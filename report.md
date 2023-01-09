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
Focusing on 20 categories, we performed SIFT feature extraction on each image,
constructing a feature pool (every feature present in every image).
We used the OpenCV SIFT implimentation, which yields 1-d vector representation
of each feature. % confirm that they are 1d?
As these features exist in the same place they are comparable, and distance can
be computed between them. Features that are sufficiently close can be grouped
together into a word (the difference between them can be viewed as similar to 
accent in human speech).


# Indexing

# Retrieving
Bag of Words representation is done by taking an image, reducing it to its SIFT features, and further representing that feature with its most closely associated word (cluster), among the $k$ candidates.

