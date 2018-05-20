# BS


## Inspiration
- https://jtsulliv.github.io/perceptron/
- https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/


## Linearly separable data

For this type of data I used `Perceptron algorithm` as recommended. 
 

### Evaluation

The provided dataset is separated by binary classes into the first and 
third quadrants. Accuracy is 100%. 


## Linearly non-sperable data

For linearly non-separable data I used SVC (Support Vector Classification) with polynomial kernel. 


### Evaluation

I achieved 100% accuracy but I believe that it is not guaranteed always. 
Although the classes are visually more or less clear, 
it depends on if the training data are representative or not. 
So if you run training multiple times, accuracy may vary because of
randomness of http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html