import gensim.models as g
import gensim.models.doc2vec as doc2vec
import codecs
from tqdm import tqdm

#parameters
model="external/models/doc2vec/doc2vec.bin"
test_docs="external/doc2vec/toy_data/test_docs.txt"
output_file="external/doc2vec/toy_data/test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)


# m.save('./test')

test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#infer test vectors
output = open(output_file, "w")
for d in tqdm(test_docs):
    output.write( " ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
output.flush()
output.close()