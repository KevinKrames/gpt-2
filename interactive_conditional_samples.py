#!/usr/bin/env python3

import argparse
import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', metavar='INPUT', type=str, default='', help='Input text to generate sentences on.')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='100M', help='Pretrained model name')
args = parser.parse_args()

def interact_model(
    model_name='100M',
    input='100M',
    seed=None,
    nsamples=3,
    batch_size=1,
    length=50,
    temperature=1.3,
    top_k=50,
    top_p=0.0
):
    """
    Interactively run the model
    :model_name=1000M : String, which model to use
    :input='' : String, input to generate on
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        #while True:
            #raw_text = input("Model prompt >>> ")
            #while not raw_text:
            #    print('Prompt should not be empty!')
            #    raw_text = input("Model prompt >>> ")
        # lines = []
        # while True:
        #     line = input("Model prompt >>> ")
        #     if line:
        #         lines.append(line)
        #     else:
        #         break
        # lines.append('')
        # raw_text = '\n'.join(lines)

        context_tokens = enc.encode(args.input)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)
