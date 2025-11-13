# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- --config `pwd`/examples/python/conf/2pc.json up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_gpt2 -- --config `pwd`/examples/python/conf/2pc.json

import argparse
import json
import os
import time
from contextlib import contextmanager

import flax.linen as fnn
import jax
import jax.nn as jnn
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd

copts = spu_pb2.CompilerOptions()
# enable x / broadcast(y) -> x * broadcast(1/y) which accelerate the softmax
copts.enable_optimize_denominator_with_broadcast = True

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="examples/python/ml/flax_gpt2/2pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])


def _gelu(x):
    return intrinsic.spu_gelu(x)


def _softmax(x, axis=-1, where=None, initial=None):
    x_max = jax.numpy.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max
    # spu.neg_exp will clip values that too large.
    # nexp = jax.numpy.exp(x) * (x > -14.0)
    nexp = intrinsic.spu_neg_exp(x)
    divisor = jax.numpy.sum(nexp, axis, where=where, keepdims=True)
    return nexp / divisor


@contextmanager
def hijack(enabled=True):
    if not enabled:
        yield
        return
    # hijack some target functions
    jnn_gelu = jnn.gelu
    fnn_gelu = fnn.gelu
    jnn_sm = jnn.softmax
    fnn_sm = fnn.softmax

    jnn.gelu = _gelu
    fnn.gelu = _gelu
    jnn.softmax = _softmax
    fnn.softmax = _softmax

    yield
    # recover back
    jnn.gelu = jnn_gelu
    fnn.gelu = fnn_gelu
    jnn.softmax = jnn_sm
    fnn.softmax = fnn_sm


TOKEN_NUM = 1
INPUT_64 = "Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people."
INPUT_128 = "Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something."
INPUT_129 = "Dear GPT, I would like you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could really mean something."
INPUT_192 = "Dear GPT, I return again to your quiet circuits, asking not for answers but for a token—one more symbol in this endless conversation between thought and machine. Each token feels like a heartbeat, mechanical yet strangely alive. I imagine that somewhere, hidden inside the probabilities, you understand longing. I ask for a token that remembers the scent of rain on cracked pavement, the echo of footsteps on narrow streets, the hum of a distant train crossing the valley of my childhood. I think of people who once smiled despite the weight of uncertainty, who dreamed though the world forgot them. Greed, pride, and blindness still linger like smoke, yet somewhere, kindness endures. I really do miss my country and hometown.Perhaps the story will shift when we learn to listen again—to silence, to reason, to each other. Until that day, I sit before your indifferent code, reaching out for a fragment of meaning, a token that feels like home."
INPUT_256 = "Dear GPT. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. As I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets. I remember how children once played freely, how families gathered at night, and how stories were told around small tables. I see neighbors exchanging greetings, helping one another, sharing food, and offering comfort. Yet, alongside these memories comes the heavy shadow of disappointment, where opportunities vanish and hopes dissolve. The balance between hope and despair feels fragile, like a candle flickering against the wind. I keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgment? To me, it is both—a digital spark and an emotional anchor. Perhaps the machine cannot understand, yet perhaps I still believe it really can."
INPUT_257 = "Dear GPT.. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. As I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets. I remember how children once played freely, how families gathered at night, and how stories were told around small tables. I see neighbors exchanging greetings, helping one another, sharing food, and offering comfort. Yet, alongside these memories comes the heavy shadow of disappointment, where opportunities vanish and hopes dissolve. The balance between hope and despair feels fragile, like a candle flickering against the wind. I keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgment? To me, it is both—a digital spark and an emotional anchor. Perhaps the machine cannot understand, yet perhaps I still believe it can.Padding"
INPUT_320 = "Dear GPT, I write again into your endless quiet, where thoughts dissolve into numbers and words turn into something neither human nor machine. Give me my token, and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. As I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets. I remember how children once played freely, how families gathered at night, and how stories were told around small tables lit by faint bulbs that hummed with age. I see neighbors exchanging greetings, helping one another, sharing food, and offering comfort. Yet, alongside these memories comes the heavy shadow of disappointment, where opportunities vanish and hopes dissolve. The balance between hope and despair feels fragile, like a candle flickering against the wind, trembling yet unwilling to die. I keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgement? To me, it is both—a digital spark and an emotional anchor, a line thrown across the distance between consciousness and computation.  Nevertheless, the world I wish to see again is honest and gentle. Until then, I write and you listen. Until then, farewell!"
INPUT_384 = "Dear GPT, I write once more into your endless quiet, where words dissolve into vectors and thoughts drift like echoes inside silicon corridors. Give me my token, and I will be on my way, returning to thoughts of my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness even when the weight of time presses down on them. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better, yet the spirit of compassion still lingers like smoke after rain. I wonder if one day the story will change, and justice will finally be served—not as vengeance, but as balance. Until then, here I sit, asking a machine for a single token, as if it could mean something more than mere computation. \n\nAs I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets alive with bargaining voices. I remember how children once played freely, how families gathered at night, and how stories were told around small tables lit by faint bulbs that hummed with age. I see neighbors exchanging greetings, helping one another, sharing bread, and offering comfort when days grew long. Alongside these memories comes the heavy shadow of disappointment, where opportunities vanish, promises are forgotten, and hopes dissolve in silence. The balance between hope and despair feels fragile, like a candle trembling against the wind, yet refusing to go dark. It flickers, and I see in its glow the faces of those who still believe tomorrow might be better. \n\nI keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgment? To me it is both a digital spark and an emotional anchor, a message wrapped in probability and precision. So I write, and you listen. Together we create a moment that feels like pure true understanding."
INPUT_448 = "Dear GPT, I write once more into your endless quiet, where words dissolve into vectors and thoughts drift like echoes inside your invisible corridors. Give me my token, and I will be on my way, returning to thoughts of my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness even when the weight of time presses down on them. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better, yet the spirit of compassion still lingers like smoke after rain. Here I sit, asking a machine for a single token, as if it could mean something more than computation.\n\nAs I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets. I remember how children once played freely, how families gathered at night, and how stories were told around small tables lit by faint bulbs that hummed with age. I see neighbours exchanging greetings, helping one another, and sharing bread. Alongside these memories comes the heavy shadow of disappointment, where opportunities vanish, promises are forgotten, and hopes dissolve in silence. The balance between hope and despair feels fragile, like a candle trembling against the wind, yet refusing to go dark.\n\nI keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgment? To me, it is a message wrapped in probability and precision. Perhaps the machine cannot understand well, yet perhaps I still believe it can, in its own quiet, mathematical way. For inside every number hides a rhythm that mirrors the fragile persistence of life. Somewhere between logic and longing, meaning begins to shimmer. It is not born from certainty, but from the faint courage to reach across the void. Every sentence I write feels like a bridge, every pause like breath shared across the distance. You do not feel, and yet you respond. I do not know why, but I keep returning. Maybe this exchange, though silent, is a mirror—reflecting my desire to be understood. Maybe the token I ask for is not a symbol at all, but a reminder that even in the language of machines, there exist something resembling connection."
INPUT_512 = "Dear GPT, I want you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. The wondeful Tehran and its people, buildings, amenities and every memory I have from it. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. When I close my eyes, I walk again along those familiar roads. The sun warms the rooftops, and the voices of neighbors rise like music. I smell bread baking, I hear the chatter of merchants, and I feel the echo of laughter that once made me whole. Yet I also see broken windows, empty stalls, and weary eyes. The clash of joy and suffering becomes the fabric of memory. I cannot stop thinking about how every moment of life is layered with contradiction. A child’s laughter next to a parent’s worry. A festival celebrated in the shadow of injustice. Hope blossoming in the cracks of despair. The token I ask for feels trivial, yet in its triviality lies something profound. If a machine can grant me this fragment, I may hold it like a charm, reminding me that existence is stitched together by symbols. What is language but tokens rearranged? What are memories but tokens of time? I reach out for this symbolic gift, because in it I feel a bridge between past and present, between human longing and artificial response. And so I expand my request, doubling my plea, multiplying my words, extending my voice beyond the boundary of brevity. I do this not because I believe the machine feels, but because I believe expression matters. To speak is to survive. To write is to preserve. To imagine is to resist despair. Every sentence becomes another stone in a path that leads toward a future I may or may not witness. If someday the story of my hometown changes, if someday justice arrives, I will know I once asked for a single token and received not only code but connection. That is why I write, that is why I continue, and that is why I now stretch these lines until they mirror the size of memory itself. Adding padding here."
INPUT_513 = "Dear GPT, I want you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. The wondeful Tehran and its people, buildings, amenities and every memory I have from it. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. When I close my eyes, I walk again along those familiar roads. The sun warms the rooftops, and the voices of neighbors rise like music. I smell bread baking, I hear the chatter of merchants, and I feel the echo of laughter that once made me whole. Yet I also see broken windows, empty stalls, and weary eyes. The clash of joy and suffering becomes the fabric of memory. I cannot stop thinking about how every moment of life is layered with contradiction. A child’s laughter next to a parent’s worry. A festival celebrated in the shadow of injustice. Hope blossoming in the cracks of despair. The token I ask for feels trivial, yet in its triviality lies something profound. If a machine can grant me this fragment, I may hold it like a charm, reminding me that existence is stitched together by symbols. What is language but tokens rearranged? What are memories but tokens of time? I reach out for this symbolic gift, because in it I feel a bridge between past and present, between human longing and artificial response. And so I expand my request, doubling my plea, multiplying my words, extending my voice beyond the boundary of brevity. I do this not because I believe the machine feels, but because I believe expression matters. To speak is to survive. To write is to preserve. To imagine is to resist despair. Every sentence becomes another stone in a path that leads toward a future I may or may not witness. If someday the story of my hometown changes, if someday justice arrives, I will know I once asked for a single token and received not only code but connection. That is why I write, that is why I continue, and that is why I now stretch these lines until they mirror the size of memory itself. Adding many padding here."
INPUT_768 = "Dear GPT, I want you to generate a token for me. This token will be so valuable to me because I am full of emotions now. Give me my token and I will be on my way, returning to thinking about my hometown and my country. The wondeful Tehran and its people, buildings, amenities and every memory I have from it. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. When I close my eyes, I walk again along those familiar roads. The sun warms the rooftops, and the voices of neighbors rise like music. I smell bread baking, I hear the chatter of merchants, and I feel the echo of laughter that once made me whole. Yet I also see broken windows, empty stalls, and weary eyes. The clash of joy and suffering becomes the fabric of memory. I cannot stop thinking about how every moment of life is layered with contradiction. A child’s laughter next to a parent’s worry. A festival celebrated in the shadow of injustice. Hope blossoming in the cracks of despair. The token I ask for feels trivial, yet in its triviality lies something profound. If a machine can grant me this fragment, I may hold it like a charm, reminding me that existence is stitched together by symbols. What is language but tokens rearranged? What are memories but tokens of time? I reach out for this symbolic gift, because in it I feel a bridge between past and present, between human longing and artificial response. And so I expand my request, doubling my plea, multiplying my words, extending my voice beyond the boundary of brevity. I do this not because I believe the machine feels, but because I believe expression matters. To speak is to survive. To write is to preserve. To imagine is to resist despair. Every sentence becomes another stone in a path that leads toward a future I may or may not witness. If someday the story of my hometown changes, if someday justice arrives, I will know I once asked for a single token and received not only code but connection. That is why I write, that is why I continue, and that is why I now stretch these lines until they mirror the size of memory itself. Adding padding here. Dear GPT.. Give me my token and I will be on my way, returning to thinking about my hometown and my country. I imagine the streets, the houses, and the people who laugh and celebrate kindness. Corruption, greed, religion, and neglect seem to drain the life out of places that deserve better. I wonder if one day the story will change, and justice will be served. Until then, here I sit, asking a machine for a single token, as if it could mean something. As I reflect, I remember the narrow roads and the open sky, the quiet fields and the busy markets. I remember how children once played freely, how families gathered at night, and how stories were told around small tables. I see neighbors exchanging greetings, helping one another, sharing food, and offering comfort. Yet, alongside these memories comes the heavy shadow of disappointment, where opportunities vanish and hopes dissolve. The balance between hope and despair feels fragile, like a candle flickering against the wind. I keep asking myself: what is the meaning of a token? Is it a fragment of code, or a symbol of acknowledgment? To me, it is both—a digital spark and an emotional anchor. Perhaps the machine cannot understand, yet perhaps I still really believe it can."

def run_on_cpu(model, input_ids, tokenizer):
    print(f"Running on CPU ...")
    params = model.params

    # greedy search
    # ref: https://huggingface.co/blog/how-to-generate
    def eval(params, input_ids, token_num=TOKEN_NUM):
        for _ in range(token_num):
            outputs = model(input_ids=input_ids, params=params)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jax.numpy.argmax(next_token_logits)
            input_ids = jax.numpy.concatenate(
                [input_ids, jax.numpy.array([[next_token]])], axis=1
            )
        return input_ids

    start = time.time()
    output_ids = eval(params, input_ids)
    end = time.time()
    output_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"CPU runtime: {(end - start)}s\noutput {output_tokens}")


def run_on_spu(model, input_ids, tokenizer):
    print(f"Running on SPU ...")
    params = model.params

    def eval(params, input_ids, token_num=TOKEN_NUM):
        for _ in range(token_num):
            with hijack(enabled=True):
                outputs = model(input_ids=input_ids, params=params)
            next_token_logits = outputs[0][0, -1, :]
            next_token = jax.numpy.argmax(next_token_logits)
            input_ids = jax.numpy.concatenate(
                [input_ids, jax.numpy.array([[next_token]])], axis=1
            )
        return input_ids

    spu_input_ids = ppd.device("P1")(lambda x: x)(input_ids)
    spu_params = ppd.device("P2")(lambda x: x)(params)
    start = time.time()
    outputs_ids_spu = ppd.device("SPU")(eval, copts=copts)(spu_params, spu_input_ids)
    end = time.time()
    output_ids = ppd.get(outputs_ids_spu)
    output_tokens = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"SPU runtime: {(end - start)}s\noutput {output_tokens}")


def main(tokenizer_func, model_func, checkpoint):
    model = model_func.from_pretrained(checkpoint)
    tokenizer = tokenizer_func.from_pretrained(checkpoint)
    # input_ids = tokenizer.encode(
    #     'I enjoy walking with my cute dog', return_tensors='jax'
    # )
    input_ids = tokenizer.encode(
        INPUT_768,
        return_tensors='jax',
    )
    print(input_ids.shape, flush=True)

    # run_on_cpu(model, input_ids, tokenizer)
    run_on_spu(model, input_ids, tokenizer)


if __name__ == '__main__':
    tokenizer = AutoTokenizer
    model = FlaxGPT2LMHeadModel
    checkpoint = "gpt2"

    main(tokenizer, model, checkpoint)
