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
