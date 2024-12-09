# __init__.py

import logging
import json
import os
import azure.functions as func

import re
import torch
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from azureml.core import Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the aspects to analyze
aspects = [
    "Product Quality",
    "Content/Performance",
    "User Experience",
    "Value for Money",
    "Customer Service",
    "Aesthetics/Design",
    "Functionality/Features",
    "Ease of Use/Accessibility",
    "Durability/Longevity",
    "Shipping and Packaging",
    "Sarcasm Detector",
    "Paid Review Detection"
]

# Azure ML Workspace authentication and model loading
def load_models():
    """
    Authenticates with Azure ML Workspace and loads the required models.
    """
    try:
        # Retrieve Azure credentials from environment variables
        tenant_id = os.environ['AZURE_TENANT_ID']
        service_principal_id = os.environ['AZURE_CLIENT_ID']
        service_principal_password = os.environ['AZURE_CLIENT_SECRET']
        subscription_id = os.environ['AZURE_SUBSCRIPTION_ID']
        resource_group = os.environ['AZURE_RESOURCE_GROUP']
        workspace_name = os.environ['AZURE_WORKSPACE_NAME']

        # Service Principal Authentication
        sp_auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password
        )

        # Connect to workspace
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=sp_auth
        )
        logger.info('Connected to Azure ML Workspace.')

        # Load models from Azure ML Model Registry
        bert_aspect_model = Model(ws, name='bert-aspect-extraction-model')
        bert_sentiment_model = Model(ws, name='bert-sentiment-classification-model')
        gpt2_model = Model(ws, name='gpt2-scoring-justification-model')

        bert_aspect_model_path = bert_aspect_model.download(exist_ok=True)
        bert_sentiment_model_path = bert_sentiment_model.download(exist_ok=True)
        gpt2_model_path = gpt2_model.download(exist_ok=True)

        logger.info('Models loaded from Azure ML Model Registry.')

        return bert_aspect_model_path, bert_sentiment_model_path, gpt2_model_path

    except Exception as e:
        logger.error('Error connecting to Azure ML Workspace or loading models: %s', e)
        raise

# Load the models and tokenizers
bert_aspect_model_path, bert_sentiment_model_path, gpt2_model_path = load_models()

# Load BERT Tokenizer and Models
try:
    # Aspect Extraction Model
    aspect_tokenizer = BertTokenizer.from_pretrained(bert_aspect_model_path)
    aspect_model = BertForTokenClassification.from_pretrained(bert_aspect_model_path)
    aspect_model.eval()

    # Sentiment Classification Model
    sentiment_tokenizer = BertTokenizer.from_pretrained(bert_sentiment_model_path)
    sentiment_model = BertForSequenceClassification.from_pretrained(bert_sentiment_model_path)
    sentiment_model.eval()

    logger.info('BERT models and tokenizers loaded successfully.')

except Exception as e:
    logger.error('Error loading BERT models: %s', e)
    raise

# Load GPT-2 Tokenizer and Model
try:
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)
    gpt2_model.eval()
    logger.info('GPT-2 model and tokenizer loaded successfully.')

except Exception as e:
    logger.error('Error loading GPT-2 model: %s', e)
    raise

def extract_aspects(review_text):
    """
    Extracts aspects from the review text using BERT for token classification.

    Parameters:
    review_text (str): The product review text.

    Returns:
    List[Dict]: A list of extracted aspects with their positions.
    """
    # Tokenization and Input Embedding
    tokens = aspect_tokenizer.tokenize(review_text)
    input_ids = aspect_tokenizer.encode(review_text, return_tensors='pt')
    attention_mask = input_ids.ne(aspect_tokenizer.pad_token_id).long()

    # Get predictions from the model
    with torch.no_grad():
        outputs = aspect_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the predicted tags
    predicted_tags = torch.argmax(logits, dim=2).squeeze().tolist()
    tags = [aspect_model.config.id2label[tag_id] for tag_id in predicted_tags]

    # Extract aspects based on predicted tags
    aspects_extracted = []
    current_aspect = ''
    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            if current_aspect:
                aspects_extracted.append(current_aspect.strip())
                current_aspect = ''
            current_aspect = token.replace('##', '')
        elif tag.startswith('I-'):
            current_aspect += ' ' + token.replace('##', '')
        else:
            if current_aspect:
                aspects_extracted.append(current_aspect.strip())
                current_aspect = ''

    if current_aspect:
        aspects_extracted.append(current_aspect.strip())

    return aspects_extracted

def classify_sentiment(aspect, sentence):
    """
    Classifies the sentiment of the given aspect within the sentence using BERT.

    Parameters:
    aspect (str): The aspect to classify sentiment for.
    sentence (str): The sentence containing the aspect.

    Returns:
    str: The sentiment label (Positive, Negative, Neutral).
    """
    # Prepare the input
    inputs = f"[CLS] {aspect_tokenizer.sep_token} {aspect} {aspect_tokenizer.sep_token} {sentence} [SEP]"
    input_ids = sentiment_tokenizer.encode(inputs, return_tensors='pt')
    attention_mask = input_ids.ne(sentiment_tokenizer.pad_token_id).long()

    # Get predictions from the model
    with torch.no_grad():
        outputs = sentiment_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the predicted sentiment
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment_label = sentiment_model.config.id2label[predicted_class]

    return sentiment_label

def score_aspect_sentiment(aspect, sentiment):
    """
    Uses GPT-2 to assign a numerical score based on the aspect and sentiment.

    Parameters:
    aspect (str): The aspect.
    sentiment (str): The sentiment associated with the aspect.

    Returns:
    int: The numerical score.
    """
    # Prepare the input prompt
    prompt = f"Aspect: {aspect}, Sentiment: {sentiment}. Score:"
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')

    # Generate the score using GPT-2
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids,
            max_length=input_ids.size(1) + 5,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract the score
    score_match = re.search(r'Score:\s*([-+]?\d+)', generated_text)
    if score_match:
        score = int(score_match.group(1))
    else:
        score = 0  # Default score if extraction fails

    return score

def generate_justification(aspect, sentiment, score, review_text):
    """
    Generates a justification for the given aspect, sentiment, and score using GPT-2.

    Parameters:
    aspect (str): The aspect.
    sentiment (str): The sentiment.
    score (int): The numerical score.
    review_text (str): The original review text.

    Returns:
    str: The generated justification.
    """
    # Prepare the input prompt
    prompt = f"Aspect: {aspect}, Sentiment: {sentiment}, Score: {score}. Review Text: {review_text} Justification:"
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')

    # Generate the justification using GPT-2
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids,
            max_length=input_ids.size(1) + 50,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract the justification
    justification = generated_text[len(prompt):].strip()

    return justification

def analyze_review(review_text):
    """
    Analyzes the review text using BERT for aspect extraction and sentiment classification,
    and GPT-2 for scoring and justification generation.

    Parameters:
    review_text (str): The product review text.

    Returns:
    list: A list of dictionaries containing analysis results for each aspect.
    """
    # Step 1: Aspect Extraction (BERT)
    aspects_extracted = extract_aspects(review_text)
    logger.info(f'Extracted Aspects: {aspects_extracted}')

    # Step 2: Sentiment Classification (BERT)
    analysis_results = []
    for aspect in aspects:
        if aspect in aspects_extracted:
            # For each aspect, extract sentences containing the aspect
            aspect_sentences = [sentence for sentence in re.split(r'(?<=[.!?]) +', review_text) if aspect.lower() in sentence.lower()]
            sentiment_scores = []
            justifications = []

            for sentence in aspect_sentences:
                # Classify sentiment
                sentiment_label = classify_sentiment(aspect, sentence)
                logger.info(f'Aspect: {aspect}, Sentiment: {sentiment_label}')

                # Step 3: Scoring (GPT-2)
                score = score_aspect_sentiment(aspect, sentiment_label)
                logger.info(f'Aspect: {aspect}, Score: {score}')

                # Step 4: Justification Generation (GPT-2)
                justification = generate_justification(aspect, sentiment_label, score, sentence)
                logger.info(f'Aspect: {aspect}, Justification: {justification}')

                sentiment_scores.append(score)
                justifications.append(justification)

            # Aggregate scores and justifications
            avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            combined_justification = " ".join(justifications)

            # Determine overall sentiment
            if avg_score >= 20:
                overall_sentiment = "Positive"
            elif avg_score <= -20:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Mixed"

            analysis_results.append({
                'Aspect': aspect,
                'Sentiment': overall_sentiment,
                'Score': round(avg_score, 2),
                'Justification': combined_justification
            })
        else:
            # Aspect not mentioned
            analysis_results.append({
                'Aspect': aspect,
                'Sentiment': "Neutral",
                'Score': 0,
                'Justification': "This aspect is not mentioned in the review."
            })

    return analysis_results

def main(req: func.HttpRequest) -> func.HttpResponse:
    logger.info('Processing request for ReviewSense analysis.')

    try:
        # Parse JSON request body
        req_body = req.get_json()
        review_text = req_body.get('review')

        if not review_text:
            logger.error('No review text provided.')
            return func.HttpResponse(
                json.dumps({'error': "Please provide a 'review' field in the JSON body."}),
                status_code=400,
                mimetype='application/json'
            )

        # Perform analysis
        analysis_results = analyze_review(review_text)

        # Construct response
        response_content = {
            'ReviewSense': "Aspect-Based Sentiment Analysis of Product Reviews",
            'Review': review_text,
            'AnalysisResults': analysis_results
        }

        return func.HttpResponse(
            json.dumps(response_content, ensure_ascii=False, indent=4),
            status_code=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error('Error processing request: %s', e)
        return func.HttpResponse(
            json.dumps({'error': 'Internal server error.'}),
            status_code=500,
            mimetype='application/json'
        )
