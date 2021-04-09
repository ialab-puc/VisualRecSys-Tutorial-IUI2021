# How to format your dataset

A recommendation dataset, like the one we used in this tutorial, requires 2 important components:

1. Interactions between users and items: Could be "likes", "purchases" or other forms of interactions
2. Images content: We're using a content-based approach, so we need the raw images to extract their features 

In this document we'll explain the expected input or format of both components.

### Representation of interactions

Usually the interactions in a dataset come in different representations, so we used a script to format the interactions in a simpler format for preprocessing script. If you want to use our scripts, make sure to use the same output format and everything should work.

We only use user-item positive feedback interactions, so each row of our dataset will contain a **user_id** column and an **item_id** column. Also, we store the **timestamp** of the interaction to split the interactions for each user that we'll use to evaluate the model (in our case, data was already divided, but sometimes you'll have to define a criteria to choose which interactions to predict). Finally, we'll leave a Boolean **evaluation** column to mark the rows that we'll use to evaluate our models.

Below, you can see and example of how our data looks in our format:

```
user_id,item_id,timestamp,evaluation
30,200501002,1105490700,False
12,200501002,1105521180,False
```

Our preprocessing script assumes this structure and creates the training samples in the appropriate format.

### Representation of images

We stored all the images in a folder in our filesystem. Each image filename was the same value as the **item_id** described in the previous step. This has the benefit that we did not require an additional mapping from filename to item_id. If your dataset requires this mapping please be careful and consistent in its usage (in particular, make sure to return the correct **item_id** in the `PreprocessingDataset` class, so your embeddings are correctly formatted).

Our scripts assume the already described structure to create the embeddings. The embeddings have the following structure stored in a `*.npy` file:

```
[
	["200501002", [0.123123, 0.13184, ...]],
	...
]
```

 This is just a structure representation. The actual object is a `numpy.ndarray` with shape `(len(image_dataset), 2)`, where each row has two elements: the **item_id** and a vector with the features of the item. The features (in general) correspond to the output of the second to last layer of a pretrained DNN network. In our script, you'll find the configuration to forward the dataset images through a pretrained ResNet50 network.

Our training scripts assume this structure and should be able to forward the data through each model using both the embedding and interactions data.

### Additional steps

Once your dataset interactions and image features are properly formatted, the provided scripts should work with no additional modifications. You'll only need to make sure that you're using your own files. If you have any problem, please contact the authors or create an issue in our repository.