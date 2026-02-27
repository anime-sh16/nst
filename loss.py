import torch

def calculate_content_loss(content_features, generated_features):
    assert content_features.shape == generated_features.shape
    
    content_loss = 0.5 * torch.sum((content_features - generated_features) ** 2)
    
    return content_loss

def get_gram_matrix(features):
    """Gram matrix of the feature map.

    Captures channel correlations independent of spatial position.
    this is what encodes texture and style in the Gatys et al. formulation.
    Normalized by spatial dimensions to keep scale consistent across layers.
    """
    batch, channels, height, width = features.shape
    features = features.reshape(batch, channels, height * width)
    
    gram_matrix = torch.matmul(features, features.transpose(1, 2))
    normalized_gram_matrix = gram_matrix / (channels * height * width)
    
    return normalized_gram_matrix

def calculate_style_layer_loss(style_features, generated_features):
    assert style_features.shape == generated_features.shape
    
    style_gram = get_gram_matrix(style_features)
    generated_gram = get_gram_matrix(generated_features)
    
    assert style_gram.shape == generated_gram.shape
    
    style_loss = 0.25 * torch.sum((style_gram - generated_gram) ** 2)
    
    return style_loss

def calculate_style_total_loss(weights, style_features, generated_features):
    """Weighted sum of style losses across layers. weights[i] controls each layer's contribution."""

    total_loss = torch.tensor(0.0, device=style_features[0].device)
    for weight, s_feat, g_feat in zip(weights, style_features, generated_features):
        layer_loss = calculate_style_layer_loss(s_feat, g_feat)
        total_loss += weight * layer_loss
        
    return total_loss