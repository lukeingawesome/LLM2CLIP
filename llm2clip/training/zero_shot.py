import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from eva_clip import get_cast_dtype, get_tokenizer
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from collections import OrderedDict

def zero_shot_classifier(model, classnames, templates, args):
    im_features = torch.load(args.imagenet_classname_feautres)
    cast_dtype = get_cast_dtype(args.precision)
    autocast = get_autocast(args.precision)
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        for i, classname in tqdm(enumerate(classnames)):
            texts = im_features[i].to(args.device, dtype=cast_dtype)
            # texts = tokenizer(texts).to(args.device)  # tokenize
            if args.distributed:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device, dtype=cast_dtype)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max': return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum': return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean': return similarity_matrix_chunk.mean(dim=1)
    else: raise ValueError("Unknown aggregate_similarity")

def zero_shot_classifier_medical(model, text_categories, l2v, args):
    """Create zero-shot classifier for medical conditions."""
    cast_dtype = get_cast_dtype(args.precision)
    autocast = get_autocast(args.precision)
    
    with torch.no_grad(), autocast():
        zeroshot_weights = {}
        for category, texts in text_categories.items():
            # Tokenize and create embeddings
            tokenized = l2v.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            embed_mask = torch.zeros_like(tokenized["attention_mask"])
            tokenized["embed_mask"] = embed_mask
            tokenized = tokenized.to(args.device)
            
            # Get text features
            text_features = l2v(tokenized)
            if args.distributed:
                text_features = model.module.text.projection(text_features.to(dtype=cast_dtype))
            else:
                text_features = model.text.projection(text_features.to(dtype=cast_dtype))
            
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
            zeroshot_weights[category] = text_features
            
    return zeroshot_weights

def run_medical(model, classifiers, dataloader, args):
    """Run zero-shot evaluation for medical conditions."""
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    results = {k: {'correct': 0, 'total': 0, 'score': [], 'gt': []} for k in classifiers.keys()}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, unit_scale=args.eval_batch_size):
            images = images.to(args.device)
            if cast_dtype is not None:
                images = images.to(dtype=cast_dtype)
            
            with autocast():
                # Get image features
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarities for each condition
                for condition, classifier in classifiers.items():
                    logits = 100. * image_features @ classifier.T
                    predictions = logits.argmax(dim=-1).cpu().numpy()  # Move to CPU before numpy conversion
                    gt = np.array([int(float(x)) for x in labels])
                    correct = (predictions == gt).astype(int)
                    
                    results[condition]['correct'] += correct.sum()
                    results[condition]['total'] += images.size(0)
                    results[condition]['score'].extend(logits[:, 0].cpu().numpy())  # Use extend instead of +=
                    results[condition]['gt'].extend(gt)

    # Calculate metrics
    for condition in results:
        results[condition]['auc'] = roc_auc_score(results[condition]['gt'], results[condition]['score'])
    
    # Calculate accuracies
    aucs = {k: v['auc'] for k, v in results.items()}
    accuracies = {k: v['correct'] / v['total'] for k, v in results.items()}
    return aucs, accuracies

def zero_shot_eval(model, l2v, data, epoch, args):
    if 'rsna' not in data and 'siim' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot rsna or siim.')

    results = {}
    # Add medical condition evaluation
    if 'rsna' in data:
        text_categories = {
            'pneumonia': ["Detected abnormalities : There is pneumonia.", "Detected abnormalities : There is no pneumonia"],
        }
        
        logging.info('Building medical zero-shot classifier')
        medical_classifier = zero_shot_classifier_medical(model, text_categories, l2v, args)
        
        logging.info('Evaluating medical conditions')
        medical_aucs, medical_accuracies = run_medical(model, medical_classifier, data['rsna'].dataloader, args)
        print('evaluating rsna')
        # Add results
        results['rsna-pneumonia-accuracy'] = medical_accuracies['pneumonia']
        results['rsna-pneumonia-auc'] = medical_aucs['pneumonia']

    if 'siim' in data:
        text_categories = {
            'pneumothorax': ['Detected abnormalities : There is pneumothorax.', 'Detected abnormalities : There is no pneumothorax.']
        }
        logging.info('Building medical zero-shot classifier')
        medical_classifier = zero_shot_classifier_medical(model, text_categories, l2v, args)
        
        logging.info('Evaluating medical conditions')
        medical_aucs, medical_accuracies = run_medical(model, medical_classifier, data['siim'].dataloader, args)
        print('evaluating siim')
        # Add results
        results['siim-pneumothorax-accuracy'] = medical_accuracies['pneumothorax']
        results['siim-pneumothorax-auc'] = medical_aucs['pneumothorax']
    if 'chexpert' in data:
        text_categories = {
            'atelectasis': ["There is atelectasis.", "There is no atelectasis."],
            'cardiomegaly': ["There is cardiomegaly.", "There is no cardiomegaly."],
            'consolidation': ["There is consolidation.", "There is no consolidation."],
            'edema': ["There is edema.", "There is no edema."],
            'pleural_effusion': ["There is pleural effusion.", "There is no pleural effusion."],
        }
        logging.info('Building medical zero-shot classifier')
        medical_classifier = zero_shot_classifier_medical(model, text_categories, l2v, args)
        
        logging.info('Evaluating medical conditions')
        medical_aucs, medical_accuracies = run_medical(model, medical_classifier, data['chexpert'].dataloader, args)
        
        # Add results
        results['chexpert-atelectasis-accuracy'] = medical_accuracies['atelectasis']
        results['chexpert-cardiomegaly-accuracy'] = medical_accuracies['cardiomegaly']
        results['chexpert-consolidation-accuracy'] = medical_accuracies['consolidation']
        results['chexpert-edema-accuracy'] = medical_accuracies['edema']
        results['chexpert-pleural_effusion-accuracy'] = medical_accuracies['pleural_effusion']
        results['chexpert-atelectasis-auc'] = medical_aucs['atelectasis']
        results['chexpert-cardiomegaly-auc'] = medical_aucs['cardiomegaly']
        results['chexpert-consolidation-auc'] = medical_aucs['consolidation']
        results['chexpert-edema-auc'] = medical_aucs['edema']
        results['chexpert-pleural_effusion-auc'] = medical_aucs['pleural_effusion']
    logging.info('Finished zero-shot imagenet.')

    return results
