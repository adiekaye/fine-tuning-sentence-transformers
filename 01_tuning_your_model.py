from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator


model = SentenceTransformer('all-mpnet-base-v2')


# Define our training examples
train_examples = [
    InputExample(texts=['I can\'t log in to my account.', 'Unable to access my account.', 'I need help with the payment process.'], label=1),
    InputExample(texts=['How do I reset my password?', 'I forgot my password. What do I do?', 'How can I upgrade my account?'], label=1),
    InputExample(texts=['My order hasn\'t arrived yet.', 'I haven\'t received my package.', 'I can\'t find the logout button.'], label=1),
    # More examples... like, a LOT more examples (you'll find out why soon)
]
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)


# Assume these are from your validation or test set
evaluator_triplets = [
    InputExample(texts=['How can I change my password?', 'What is the process for password change?', 'My package is delayed.']),
    InputExample(texts=['I can\'t log in to my account.', 'Unable to access my account.', 'I want to change my shipping address.']),
    # Put in as many triplets as you like...
]

# Define the evaluator
evaluator = TripletEvaluator.from_input_examples(evaluator_triplets, name='my_evaluator')

# Define the loss function
train_loss = losses.TripletLoss(model=model)

# Fine-tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=500)

# Save the model
model.save("tuned_models/fine_tuned_model")
