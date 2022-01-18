import matplotlib.pyplot as plt

import numpy as np
import tensorflow.keras.preprocessing.image as img
import tensorflow as tf
import time
plt.rcParams["figure.figsize"] = (16,32)

van_gog=img.load_img(r"C:/Users/USER/Desktop/vangog.jpg")
mona=img.load_img(r"C:/Users/USER/Desktop/mona.jpg")
m=np.array(mona)
van_gog=tf.image.resize(van_gog,[954,640])
van_gog=img.img_to_array(van_gog)
van_gog=van_gog.astype(np.float32)
mona=img.img_to_array(mona)
mona=mona.astype(np.float32)
vgg=tf.keras.applications.vgg19.VGG19(include_top=False,weights="imagenet")
content_layers=['block5_conv2']
style_layers=['block1_conv1',
             'block2_conv1',
             'block3_conv1',
             'block4_conv1',
             'block5_conv1']

contentoutp=[vgg.get_layer(j).output for j in content_layers]
styleoutp=[vgg.get_layer(j).output for j in style_layers]
model_outputs=styleoutp+contentoutp
model=tf.keras.Model(vgg.inputs,model_outputs)
for u in model.layers:
    u.trainable=False
h=model(van_gog[np.newaxis])
style_features=[style_layer[0] for style_layer in h[:5]]
content_features=[content_layer[0] for content_layer in h[5:]]
class style_transfer():
    def content_loss(self,pred,target):
        return tf.reduce_sum(tf.square(pred-target))
    def get_model(self,image):
        model_outputs=model(image)
        return model_outputs[:5],model.outputs[5:]
    def gram_matrix(self,tensor):
        channels=tensor.shape[-1]
        a=tf.reshape(tensor,[-1,channels])
        gram_matrix=tf.matmul(a,a,transpose_a=True)
        return gram_matrix
    def style_loss(self,base,target):
        base_style=self.gram_matrix(base)
        target_style=self.gram_matrix(target)
        return tf.reduce_mean(tf.square(base_style-target_style))
    def compute_loss(self,init_image,target_image,weights):
        input_layers = model(init_image[np.newaxis])
        target_layers = model(target_image[np.newaxis])
        inp_style_features = [style_layer[0] for style_layer in input_layers[:5]]
        inp_content_features = [content_layer[0] for content_layer in input_layers[5:]]
        trg_style_features = [style_layer[0] for style_layer in target_layers[:5]]
        trg_content_features = [content_layer[0] for content_layer in target_layers[5:]]
        style_weight=weights[0]
        content_weight=weights[1]
        style_weight_per_layer=1/len(style_layers)
        content_weight_per_layer=1/len(content_layers)
        style_score=0
        content_score=0
        for inp,trg in zip(inp_style_features,trg_style_features):
            style_score+=style_weight_per_layer * self.style_loss(inp,trg)
        for inp, trg in zip(inp_content_features, trg_content_features):
            content_score += content_weight_per_layer * self.content_loss(inp, trg)
        style_score*=style_weight
        content_score*=content_weight
        all_loss=style_score+content_score
        return all_loss,style_score,content_score
    def compute_grad(self,cfg):
        with tf.GradientTape(persistent=True) as tape:
            all_loss=self.compute_loss(**cfg)
        loss=all_loss[0]
        return tape.gradient(loss,cfg["init_image"]),all_loss

    def train(self,init_image,target_image,weights,num_iterations=1000 ):
        n_iterations=1
        opt=tf.keras.optimizers.Adam(learning_rate=10)
        init_image=tf.Variable(init_image,dtype=tf.float32)

        cfg={"init_image":init_image,
             "target_image":target_image,
             "weights":weights}
        start_time=time.time()
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means
        best_loss=float("inf")
        best_image=0
        images_per_iterations=[]
        for train_step in range(num_iterations):
            grads,loss=self.compute_grad(cfg)
            all_loss,style_loss,content_loss=loss
            opt.apply_gradients([(grads,init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            n_iterations+=1
            if np.array(loss[0])<best_loss:
                best_loss=np.array(loss[0])
                best_img=init_image.numpy()
            if n_iterations%100==0:
                end_time=time.time()
                images_per_iterations.append(init_image)
                print("iterations:{}".format(n_iterations))
                print("loss:{} content_loss:{} style_loss:{}".format(all_loss,content_loss,style_loss))
                print("time:{}".format(end_time-start_time))
                start_time=end_time
        self.init_image=init_image
        self.best_img=best_img
        self.images=np.array(images_per_iterations)

model=style_transfer()
model.train(mona,van_gog,weights=[1e-2,1e3])
tr=np.array(model.init_image)
for t in range(1,11):
    plt.subplot(2,5,t)
    j=model.images[t-1]
    plt.imshow(j.astype(int))
plt.show()
plt.imshow(model.best_img.astype(int))
plt.show()
