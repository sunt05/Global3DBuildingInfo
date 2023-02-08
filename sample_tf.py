import json
import tensorflow as tf


dta_sample = tf.concat([tf.ones(shape=[1, 20, 20, 7])*i for i in range(0, 8)], axis=0)
dta_numpy = dta_sample.numpy().tolist()
print(len(dta_numpy))

dta_ref = {"args0": dta_numpy}

with open("testSample/sample.json", "w") as outfile:
    outfile.write(json.dumps(dta_numpy))