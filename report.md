# introduction
Similar to how texts can, for example, be classified into spam or not spam,
in the specific case of e-mail, based exclusively on the knowledge of the
presence of particular words (disregarding order), so too can images.
In the case of words images would refer to particular patterns, while
disregarding there order means, representing the image as the set of patterns
it includes—not where they are located.
Detecting (and defining) ``particular pattern'' is, however, not as intuitive
as with written sentences.
One algorithm for word extraction from images is
scale-invariant feature transform (SIFT).
SIFT features are extracted by convoling an images with Gausian filters of
different scales, % Describe this a bit better.
Afterwards, a vocabulary is contructed by clustering the observered features
into $n$ words.

# Codebook generation
Focusing on XXXX categories, we split those into train and test sets each (80/20).
On the train set we performed SIFT feature extraction on each image,
constructing a feature matrix M (every feature present in every image).
We used the OpenCV SIFT implimentation, which yields 1d vector representation
of each feature, which were stacked to create M. % confirm that they are 1d?
As these features exist in the same space they are comparable, and distance can
be computed between them. Features that are sufficiently close can be clustered
together into a word (the difference between them can be viewed as similar to 
accent in human speech). We used scikit-learns $k$-means cluster, begging the
hyper parameter $k$ (which in this case comes to represent covabulary size).
We ran our code with three different values of $k$: 250, 500, and 1000.





# Indexing
For our indexing step, we first, for each image in the test set, represented
that image as its SIFT features. Each sift feature was then assigned to the word
representing its cluster.

# Retrieving
Bag of Words representation is done by taking an image, reducing it to its SIFT features, and further representing that feature with its most closely associated word (cluster), among the $k$ candidates.

