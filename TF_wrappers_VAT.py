import tensorflow as tf







# Definition of network
def Unet(x, inputdepth, num_classes, size4net1, size4net2, re=None, getter=None, is_training_flag=False):
    with tf.variable_scope('Model', reuse=re, custom_getter=getter):

        x_shape = tf.shape(x)
        #output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2]) # [BS doubleheight doubl width  halvedepth]
        resize_method=tf.image.ResizeMethod.BILINEAR
        resize_method2=tf.image.ResizeMethod.BILINEAR
        resize_method3=tf.image.ResizeMethod.NEAREST_NEIGHBOR

        stvs=0.01
        depthstart=64
        size_tens=(tf.cast(tf.identity(size4net1), tf.int32), tf.cast(tf.identity(size4net2), tf.int32))
        size_tens2=(tf.cast(tf.identity(size4net1/4), tf.int32), tf.cast(tf.identity(size4net2/4), tf.int32))
       
        x=tf.image.resize_images(x, size_tens, method=resize_method)

        conv1a = conv_layer_2d(x, inputdepth, depthstart, stvs,scopename="conv1a") 
        conv1b = conv_layer_2d(conv1a, depthstart, depthstart, stvs, scopename="conv1b")
        pooled1 = maxpool2d(conv1b,scopename="pooled1")
        
        conv2a = conv_layer_2d(pooled1, depthstart, depthstart*2, stvs,scopename="conv2a") 
        conv2b = conv_layer_2d(conv2a, depthstart*2, depthstart*2, stvs, scopename="conv2b")
        pooled2 = maxpool2d(conv2b,scopename="pooled2")
    
        conv3a = conv_layer_2d(pooled2, depthstart*2, depthstart*4, stvs,scopename="conv3a") 
        conv3b = conv_layer_2d(conv3a, depthstart*4, depthstart*4, stvs, scopename="conv3b")
        conv3b = conventional_dropout(conv3b, train_flag=is_training_flag, scopename="conventional_dropout3")
        pooled3 = maxpool2d(conv3b,scopename="pooled3")
    
        conv4a = conv_layer_2d(pooled3, depthstart*4, depthstart*8, stvs,scopename="conv4a") 
        conv4b = conv_layer_2d(conv4a, depthstart*8, depthstart*8, stvs, scopename="conv4b")
        conv4b = conventional_dropout(conv4b, train_flag=is_training_flag, scopename="conventional_dropout1")
        pooled4 = maxpool2d(conv4b,scopename="pooled4")
    
        # THROAT
        conv5a = ASPP(pooled4, depthstart*8,  depthstart*16, depthstart, stvs, scopename='ASPP1_')
        conv5b = ASPP(conv5a, depthstart*16,  depthstart*16, depthstart, stvs, scopename='ASPP2_')
        conv5c = ASPP(conv5b, depthstart*16,  depthstart*8, depthstart, stvs, scopename='ASPP3_')
        conv5c = conventional_dropout(conv5c, train_flag=is_training_flag, scopename="conventional_dropout2")

        # Upsample 1, takes it halfway
        upconv1=tf.image.resize_images(conv5c, size_tens2, method=resize_method2)
        conv6a = conv_layer_2d(upconv1, depthstart*8, depthstart*4, stvs,scopename="conv6a")
        conc1 = conv3b+conv6a#concatenate(conv3b, conv6a, scopename="concat1")
        conv6b = conv_layer_2d(conc1, depthstart*4, depthstart*4, stvs,scopename="conv6b")

        deepsup1=conv_layer_2d(conv6b, depthstart*4, 64, stvs, scopename='DeepSup1')
        deepsup1=conv_layer_2d(deepsup1, 64, 64, stvs, scopename='DeepSup1p1')
        deepsup1 = tf.nn.softmax(tf.image.resize_images(linear_conv_layer_2d(deepsup1, 64, num_classes, stvs, f_size=1, scopename='DeepSup1p2'), (x_shape[1],x_shape[2]), method=resize_method3) )
        
        # Upsample 2
        upconv2=tf.image.resize_images(conv6b, size_tens, method=resize_method2)
        conv7a = conv_layer_2d(upconv2, depthstart*4, depthstart, stvs, scopename="conv7a")
        conc2=conv1b+conv7a#concatenate(conv1b, conv7a, scopename="concat2")
        conv7b = conv_layer_2d(conc2, depthstart, depthstart, stvs, scopename="conv7b")
        
        # Reduce depth to num_classes
        with tf.name_scope("Logs"):
            s=''
            w=tf.get_variable(name='Logits_weights', initializer=tf.random_normal([1, 1, depthstart, num_classes],stddev=stvs), trainable=True)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
            
            b=tf.get_variable(name='Logits_bias', initializer=tf.ones([num_classes]), trainable=True)
    
            loglayer = tf.nn.conv2d(conv7b, w, strides=[1, 1, 1, 1], padding='SAME')
            loglayer = tf.nn.bias_add(loglayer,b)
            
            loglayer=tf.image.resize_images(loglayer, (x_shape[1],x_shape[2]), method=resize_method) # Upsample back up

        if re==None:
            return loglayer, deepsup1
        else:
            return loglayer












def PRELU(_x,scopename='ALPH'):
  alphas = tf.get_variable('alpha_'+scopename, _x.get_shape()[-1], initializer=tf.constant_initializer(0.0),dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg



# Layer wrappers
def linear_conv_layer_2d(inputs, channels_in, channels_out, stvs=0.01, strides=1, scopename="Conv", f_size=3, use_bias=True, dil=1):
    with tf.name_scope(scopename):
        s=''
        weightname=(scopename,'_weightz')
        biasname=(scopename,'_biaz')
        
        w=tf.get_variable(name=s.join(weightname), initializer=tf.random_normal([f_size, f_size, channels_in, channels_out],stddev=stvs))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME',dilations=[1,dil,dil,1])
        
        if use_bias==True:
            b=tf.get_variable(name=s.join(biasname), initializer=tf.random_normal([channels_out],stddev=stvs))
            x = tf.nn.bias_add(x, b)
        
        return x


# Layer wrappers
def conv_layer_2d(inputs, channels_in, channels_out, stvs=0.01, strides=1, scopename="Conv"):
    with tf.name_scope(scopename):
        s=''
        weightname=(scopename,'_weights')
        biasname=(scopename,'_bias')
        s_name=(scopename,'_scale')
        b_name=(scopename,'_shift')
		
        w=tf.get_variable(name=s.join(weightname), initializer=tf.random_normal([3, 3, channels_in, channels_out],stddev=stvs), trainable=True) # ,name=s.join(weightname)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)

        
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')

        x=IN_2D(x, s.join(s_name), s.join(b_name))

        return(PRELU(x,scope_name=scopename))
		
		
def IN_2D(x, s_name, b_name):
    epsilon = 1e-3;
    scale = tf.get_variable(name=s_name, initializer=tf.ones([x.get_shape()[-1]]), trainable=True)
    beta = tf.get_variable(name=b_name, initializer=tf.zeros([x.get_shape()[-1]]), trainable=True)
    batch_mean, batch_var = tf.nn.moments(x,[1,2],keep_dims=True)
    x=tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)
    return x


def PRELU(_x,scope_name='ALPH'):
  alphas = tf.get_variable('alpha_'+scope_name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.01),dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def ASPP(x, c_in, c_out, d_start, stvs, scopename='ASPP'):

    x1=linear_conv_layer_2d(x, c_in, int(c_out/4), stvs, scopename=scopename+'1')
    x2=linear_conv_layer_2d(x, c_in, int(c_out/4), stvs, dil=2, scopename=scopename+'2')
    x3=linear_conv_layer_2d(x, c_in, int(c_out/4), stvs, dil=4, scopename=scopename+'3')
    x4=linear_conv_layer_2d(x, c_in, int(c_out/4), stvs, dil=8, scopename=scopename+'4')

    x=tf.concat([x1,x2,x3,x4], axis=-1)
    x=conv_layer_2d(x, int(c_out/4)*4, c_out, stvs, scopename=scopename+'5')
    
    return x

def maxpool2d(x, k=2, scopename="Pool"):
    with tf.name_scope(scopename):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')

def upconv2d(x, channels_in, channels_out, stvs, stride=2, scopename="Upconv"):
    with tf.name_scope(scopename):
        s=''
        weightname=(scopename,'_weights')
        w=tf.get_variable(name=s.join(weightname),initializer=tf.random_normal([2, 2, channels_out, channels_in],stddev=stvs), trainable=True)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, channels_out]) # [BS doubleheight doubl width  halvedepth]
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def concatenate(in1, in2, scopename="Concat"):
    with tf.name_scope(scopename):
        return tf.concat([in1, in2], 3) 
    
def conventional_dropout(in_tensor, train_flag, scopename="conventional_dropout"):
    with tf.name_scope(scopename):
        prob_drop=0.6 #Parameter for sampling Bernoulli distribution
        tensor_shape=tf.shape(in_tensor) #Need shape of input
        dims=tf.unstack(tensor_shape) 
        sample_shape=tf.stack([tensor_shape[0], tensor_shape[1], tensor_shape[2],  tensor_shape[3]], 0)

        #sample=prob_keep
        #sample+=tf.random_uniform(sample_shape) # Adding the probability of keeping to a number between 0 and 1
        #drop_tensor=tf.floor(sample) # Taking the floor of each element returns either 0 or 1
		
        #tensor_out=tf.nn.dropout(in_tensor, rate=p, training=train_flag, noise_shap=sample_shape)
        tensor_out=tf.layers.dropout(in_tensor, rate=0.3, training=train_flag, noise_shape=sample_shape)

        return tensor_out
		
		
		
		
		

def SE_2D(x, num_channels, scope_name='SE_2D_Generic', r=8): # https://arxiv.org/abs/1709.01507

    c1=int(num_channels)
    c2=int(num_channels/r)
	
    # X is [b,H,W,n]
    squeeze=tf.reduce_mean(x, axis=[1, 2], keep_dims=False) # [b, c1] 
    w1=tf.get_variable(name=scope_name + '_SE_2D_W1', shape=[c1,c2],dtype=tf.float32) 
    w2=tf.get_variable(name=scope_name + '_SE_2D_W2', shape=[c2,c1],dtype=tf.float32)
    excite=tf.nn.sigmoid(tf.matmul(tf.nn.relu(tf.matmul(squeeze,w1)),w2)) # [b, c1]
    excite=tf.expand_dims(excite, axis=1)
    excite=tf.expand_dims(excite, axis=1)

    y=x*excite;
    
    return y
	
	
	
	
	
	
def res_BASE_2D(x, num_channels_in, num_channels_out, scope_name='res_2d_base'):

    x=IN_2D(x, s_name=scope_name+'_1_s', b_name=scope_name+'_1_b')
    x=PRELU(x, scope_name=scope_name+'_1')
    x=linear_conv_layer_2d(x, num_channels_in, num_channels_in, f_size=3, use_bias=True, scopename=scope_name+'_conv1')
	
    x=IN_2D(x, s_name=scope_name+'_2_s', b_name=scope_name+'_2_b')
    x=PRELU(x, scope_name=scope_name+'_2')
    x=linear_conv_layer_2d(x, num_channels_in, num_channels_out, f_size=3, use_bias=True, scopename=scope_name+'_conv2')

    return x


	
def res_2D(x, num_channels_in, num_channels_out, scope_name='res_2d_base'):

    # Call base
    y=res_BASE_2D(x, num_channels_in, num_channels_out, scope_name=scope_name+'_resbase')
    if num_channels_in!=num_channels_out: # Projection
        x=linear_conv_layer_2d(x, num_channels_in, num_channels_out, f_size=1, use_bias=False, scopename=scope_name+'_projection')

    return x+y
	
	
	
def SE_res_2D(x, num_channels_in, num_channels_out, scope_name='SE_res_2d'):

    # Call base
    y=res_BASE_2D(x, num_channels_in, num_channels_out, scope_name=scope_name+'_resbase')
    if num_channels_in!=num_channels_out: # Projection
        x=linear_conv_layer_2d(x, num_channels_in, num_channels_out, f_size=1, use_bias=False, scopename=scope_name+'_projection')
	
	# Call SE
    y=SE_2D(y, num_channels_out, scope_name=scope_name+'SE2D')

    return y+x
	

def get_one_hot(image, num_classes): # Image is an [m  n 1] matrix
    
    # Make sure ground truth image is either 0 or 1, not 0 or 255
    #if np.amax(image>0):
    # image=np.divide(image,255)
    #image=image.astype(int)
    
    b=np.zeros((image.shape[0],image.shape[1],num_classes))
    for kk in range(image.shape[0]):
        for jj in range(image.shape[1]):
            #for ii in range(image.shape[2]):
            b[kk, jj, int(image[kk,jj,0])]=1
    
    # b is the one-hot version of image with dimensions [m n D num_classes]
    return b 
		
# Generalized dice loss function
def dice_loss(logits, onehot_labels,  weight_vec):
    with tf.name_scope("Dice_Loss"):

        # Weight-vec weights each class differently to account for unbalanced classes
        
        # loss_mask weights each sample based on INDICATOR{Image has ground}
        eps = 1e-5
        prediction = tf.nn.softmax(logits)
        intersection = tf.reduce_sum(weight_vec * prediction * onehot_labels)
        # Sum over dims and batch first, then apply class_specific weights, then sum over the rest
        union =  eps + tf.reduce_sum(weight_vec*(tf.reduce_sum(prediction, axis=[0,1,2], keep_dims=True) + tf.reduce_sum(onehot_labels,axis=[0,1,2], keep_dims=True)))
        dice_loss = -(2 * intersection/ (union))

        return dice_loss


def augment_teacher(y_aug, x_sample, y_sample, t1, t2):

    # Apply same geometric transforms to teacher prediction to be consistent with augmented supervised stream
    #y_aug = tf.contrib.image.transform(y_aug, t1, interpolation="NEAREST",name=None)     
    #y_aug = tf.contrib.image.transform(y_aug, t2, interpolation="NEAREST",name=None) 
    y_aug=tf.image.resize_image_with_crop_or_pad(y_aug,tf.squeeze(y_sample),tf.squeeze(x_sample)) # Crop or pad

    return y_aug

		
		
def augmentation(in_sup_aug, ground_sup_aug, in_uns_aug, num_classes):

    if 1==1:#is_training==True:


        # SUPERVISED STREAM
        in_sup_aug= tf.image.random_brightness(in_sup_aug, max_delta=0.7) # Brightness
        in_sup_aug = tf.image.random_contrast(in_sup_aug, 0.5,1.5) # Brightness
        G_noise=tf.random_normal(tf.shape(in_sup_aug), mean=0, stddev=0.05)
        in_sup_aug=in_sup_aug+G_noise

        flip_stack=tf.concat([in_sup_aug,ground_sup_aug], axis=0) # Flip
        #flip_stack, t1, t2=tf_shear(flip_stack)

        param1 = tf.placeholder(tf.float32)
        param2 = tf.placeholder(tf.float32)
        x_shape = tf.shape(in_sup_aug)
        y_sample=tf.to_int32((tf.random_uniform([1], minval=0, maxval=param1)+param2)*tf.to_float(x_shape[1]))
        x_sample=tf.to_int32((tf.random_uniform([1], minval=0, maxval=param1)+param2)*tf.to_float(x_shape[2]))

        in_sup_aug, ground_sup_aug =tf.split( tf.squeeze(flip_stack, axis=-1), 2, axis=0) # Split back up
        ground_sup_aug=tf.to_int32(ground_sup_aug)

        in_sup_aug=tf.expand_dims(in_sup_aug, axis=-1)
        ground_sup_aug=tf.expand_dims(ground_sup_aug, axis=-1)

        in_sup_aug=tf.image.resize_image_with_crop_or_pad(in_sup_aug,tf.squeeze(y_sample),tf.squeeze(x_sample)) # Crop or pad
        ground_sup_aug=tf.image.resize_image_with_crop_or_pad(ground_sup_aug,tf.squeeze(y_sample),tf.squeeze(x_sample)) # Crop or pad

        ground_sup_aug=tf.one_hot(tf.squeeze(ground_sup_aug, axis=-1), int(num_classes)) 


    if True:
        # UNSUPERVISED STREAM
        in_uns_aug = tf.image.random_brightness(in_uns_aug, max_delta=0.7)
        in_uns_aug = tf.image.random_contrast(in_uns_aug, 0.5,1.5)
        G_noise2=tf.random_normal(tf.shape(in_uns_aug), mean=0, stddev=0.05)
        t1=[]; t2=[];#in_uns_aug, t1, t2=tf_shear(in_uns_aug)
        in_uns_aug=tf.image.resize_image_with_crop_or_pad(in_uns_aug,tf.squeeze(y_sample),tf.squeeze(x_sample)) # Crop or pad

    return in_sup_aug, ground_sup_aug, in_uns_aug, param1, param2, x_sample, y_sample, t1, t2



		
def LSM(inputs):

    input=inputs-tf.reduce_max(inputs,3, keep_dims=True) 
	
    return inputs - tf.log(tf.reduce_sum(tf.exp(inputs), 3, keep_dims=True))
	
def KL_div(y1, y2): # First argument is original logits, second is from perturbed input

    q=tf.nn.softmax(y1)
    qlogp=tf.reduce_mean(tf.reduce_sum(q*LSM(y1),3))
    qlogq=tf.reduce_mean(tf.reduce_sum(q*LSM(y2),3))

    return qlogq - qlogp
		
def VAT_loss(X_total, logits_total, input_depth, num_classes, epsilon, size4net1, size4net2, is_training_flag=False): # Original VAT code at: https://github.com/takerum/vat_tf/blob/master/vat.py

    # Get virtual adversarial noise
    d = tf.random_normal(shape=tf.shape(X_total))
    d = d / (1e-12 + tf.reduce_max(tf.abs(d), [1,2,3], keep_dims=True)) # Reduce magnitude
    d = d / tf.sqrt(1e-8 + tf.reduce_sum(tf.pow(d, 2.0), [1,2,3], keep_dims=True)) # Normalize
    XI=1e-12
    d=XI*d

    logs_perturbed_initial=Unet(X_total + d, input_depth, num_classes, size4net1, size4net2, re=True, is_training_flag=False)

    KLD_initial=KL_div(logits_total, logs_perturbed_initial)
    d=tf.stop_gradient(tf.gradients(KLD_initial, [d], aggregation_method=2)[0])
	
    d = d / (1e-12 + tf.reduce_max(tf.abs(d), [1,2,3], keep_dims=True))
    d = d / tf.sqrt(1e-6 + tf.reduce_sum(tf.pow(d, 2.0), [1,2,3], keep_dims=True))

    eps_expand=tf.broadcast_to(epsilon, [1,1,1,1])

    r_vadv =  eps_expand * d


    # Perturb input by optimal noise
    X_perturbed=X_total+r_vadv

	
    # Get logits of perturbed example
    logits_perturbed=Unet(X_total+r_vadv, input_depth, num_classes, size4net1, size4net2,  re=True, is_training_flag=False)

    #loss_op_V=KL_div(tf.stop_gradient(logits_total), logits_perturbed)
    loss_op_V=tf.reduce_mean(tf.square(tf.nn.softmax(logits_total)-tf.nn.softmax(logits_perturbed)))

    return tf.identity(loss_op_V), X_perturbed, logits_perturbed
		



		
		
		
