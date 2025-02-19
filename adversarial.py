import tensorflow as tf

def create_adversarial_pattern(model, image, true_labels, epsilon=0.01):
    """
    Generate an adversarial image for a single input image.
    
    Args:
        model: A trained model that outputs a list of predictions (one per CAPTCHA character).
        image: A tensor of shape (1, height, width, channels).
        true_labels: List of true one-hot encoded labels for each character.
                     Each element should be a tensor of shape (1, num_chars).
        epsilon: Magnitude of the adversarial perturbation.
    
    Returns:
        adversarial_image: A tensor with adversarial noise applied.
    """
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        predictions = model(image)
        loss = 0
        for pred, true in zip(predictions, true_labels):
            loss += tf.keras.losses.categorical_crossentropy(true, pred)
    
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversarial_image = image + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)
    return adversarial_image
