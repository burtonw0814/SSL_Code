
import tensorflow as tf
import numpy as np
import os
import cv2

from convnet import *
from TF_wrappers import *
from get_batch import *





class CNN():

    def __init__(self, input_depth, num_classes, ID_lis):

        self.size4net1 = tf.placeholder(tf.int32)
        self.size4net2 = tf.placeholder(tf.int32)	
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Define placeholders
        with tf.name_scope("Input"):
            self.X_sup = tf.placeholder(tf.float32, [None, None, None])
            self.X_uns = tf.placeholder(tf.float32, [None, None, None])
        with tf.name_scope("Ground_Truth"):
            self.Y_sup = tf.placeholder(tf.float32, [None, None, None])

        
        # Identities
        in_sup=tf.identity(self.X_sup)
        in_uns=tf.identity(self.X_uns)
        ground_sup=tf.identity(self.Y_sup)


        # Augmented stream
        in_sup_aug = tf.expand_dims(in_sup, axis=-1)
        in_uns_aug = tf.expand_dims(in_uns, axis=-1)
        ground_sup_aug = tf.expand_dims(ground_sup, axis=-1)
        in_sup_aug, ground_sup_aug, in_uns_aug, self.param1, self.param2, x_sample, y_sample, t1, t2=augmentation(in_sup_aug, ground_sup_aug, in_uns_aug, num_classes)


        # Non-augmented stream
        in_sup=tf.expand_dims(in_sup, axis=-1)
        in_uns=tf.expand_dims(in_uns, axis=-1)
        ground_sup=tf.one_hot(tf.to_int32(ground_sup), int(num_classes)) 



        # Define flow graph --> Supervised Augmented Stream
        with tf.name_scope("Sup_Aug"):
            log_sup_aug, deepsup1 = Unet(in_sup_aug, input_depth, num_classes, self.size4net1, self.size4net2, is_training_flag=self.is_training)
            self.prediction_sup_aug = tf.nn.softmax(log_sup_aug)

        # Define flow graph --> UNSupervised Augmented stream
        with tf.name_scope("uns_Aug"):
            log_uns_aug = Unet(in_uns_aug, input_depth, num_classes, self.size4net1, self.size4net2, re=True, is_training_flag=self.is_training)
            self.prediction_uns_aug = tf.nn.softmax(log_uns_aug)



        ########################### EMA STUFF ############################
        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var
        
        self.decay_PL=tf.placeholder(tf.float32)#0.99
        graph_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model')
        ema = tf.train.ExponentialMovingAverage(self.decay_PL)
        ema_op = ema.apply(graph_vars)	
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op )
        graph_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ###############################################



        with tf.name_scope("Sup_Ema"):	
            self.pred_ema_sup_aug=tf.stop_gradient(tf.nn.softmax(Unet(in_sup_aug, input_depth, num_classes, self.size4net1, self.size4net2, re=True, getter=ema_getter, is_training_flag=self.is_training))) 

        with tf.name_scope("Uns_Ema"):
            self.pred_ema_uns=tf.stop_gradient(tf.nn.softmax(Unet(in_uns, input_depth, num_classes, self.size4net1, self.size4net2, re=True, getter=ema_getter, is_training_flag=self.is_training)))
        # Apply augmentation to output to compare to input
        self.pred_ema_uns_AUG =  augment_teacher(self.pred_ema_uns, x_sample, y_sample, t1, t2);

        # Define Loss
        with tf.name_scope("Loss"):
            self.deep_sup_weight = tf.placeholder(tf.float32)
            self.weight_vec = tf.placeholder(tf.float32, [1, 1, 1, num_classes])
            self.consistency_weight = tf.placeholder(tf.float32)

            deep_loss=self.deep_sup_weight*dice_loss(deepsup1, ground_sup_aug, self.weight_vec)
            loss_op_dice_sup=dice_loss(log_sup_aug, ground_sup_aug, self.weight_vec)
		    
            consistency_loss_sup=self.consistency_weight*tf.reduce_mean(tf.square(self.pred_ema_sup_aug-self.prediction_sup_aug))
            consistency_loss_uns=self.consistency_weight*tf.reduce_mean(tf.square(self.pred_ema_uns_AUG-self.prediction_uns_aug))
            
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.000001)
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            
            loss_op = loss_op_dice_sup + consistency_loss_uns + consistency_loss_sup + reg_term + deep_loss
            
            tf.summary.scalar("Loss",loss_op)
            tf.summary.scalar("Loss_Dice_Sup",loss_op_dice_sup)
            tf.summary.scalar("Reg_Pen", reg_term)
            tf.summary.scalar("Loss_Consistency_Sup", consistency_loss_sup)
            tf.summary.scalar("Loss_Consistency_Uns", consistency_loss_uns)

            tf.summary.scalar("Weighted_Deep_Loss", deep_loss)
	    

        # Consistency check:
        check_c1=tf.reduce_sum(tf.square(self.pred_ema_sup_aug-self.prediction_sup_aug))
        tf.summary.scalar("consistency_difference_sup", check_c1)
        check_c2=tf.reduce_sum(tf.square(self.pred_ema_uns_AUG-self.prediction_uns_aug))
        tf.summary.scalar("consistency_difference_uns", check_c2)



        # Define optimizer
        with tf.name_scope("Optimizer"):
            global_step = tf.Variable(0, trainable=False)
            self.L_rate=tf.placeholder(tf.float32)


            opt = tf.train.RMSPropOptimizer(learning_rate=self.L_rate)
            t_vars = tf.trainable_variables()
            accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()),trainable=False) for tv in t_vars]                                        
            self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

            # compute gradients for a batch
            batch_grads_vars = opt.compute_gradients(loss_op, t_vars)
            self.accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in enumerate(batch_grads_vars) if batch_grad_var[0] is not None]

            with tf.control_dependencies(graph_update_ops):
                self.train_op = opt.apply_gradients([(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in enumerate(batch_grads_vars)])

            tf.summary.scalar("Step", global_step)
            tf.summary.scalar("Learning_Rate", self.L_rate )


        with tf.name_scope("A_Metrics"):
            # Dice similarity coefficient
            eps = 1e-5
            intersection = tf.reduce_sum(self.prediction_sup_aug * ground_sup_aug)
            union =  eps + tf.reduce_sum(self.prediction_sup_aug) + tf.reduce_sum(ground_sup_aug)
            self.dice = (2 * intersection/ (union))
            tf.summary.scalar("Dice_Similarity_Coefficient_Supervised", self.dice)

            # Dice similarity coefficient
            eps = 1e-5
            intersection_EMA = tf.reduce_sum(self.pred_ema_sup_aug * ground_sup_aug)
            union_EMA =  eps + tf.reduce_sum(self.pred_ema_sup_aug) + tf.reduce_sum(ground_sup_aug)
            self.dice_EMA = (2 * intersection_EMA/ (union_EMA))
            tf.summary.scalar("Dice_Similarity_Coefficient_Supervised_EMA", self.dice_EMA)


            # Images for TensorBoard
            tf.summary.image('Predict_Sup_Aug', tf.cast(tf.expand_dims(tf.argmax(log_sup_aug, axis=3), 3),tf.float32), input_depth)
            tf.summary.image('Ground_Sup_Aug', tf.cast(tf.expand_dims(tf.argmax(ground_sup_aug, axis=3), 3),tf.float32), input_depth)
            tf.summary.image('Slice_Sup_Aug', tf.cast(in_sup_aug, tf.float32), input_depth) 

            # Images for TensorBoard
            #tf.summary.image('Predict_Sup', tf.cast(tf.expand_dims(tf.argmax(log_sup, axis=3), 3),tf.float32), input_depth)
            #tf.summary.image('Ground_Sup', tf.cast(tf.expand_dims(tf.argmax(ground_sup, axis=3), 3),tf.float32), input_depth)
            #tf.summary.image('Slice_Sup', tf.cast(in_sup, tf.float32), input_depth) 

            # Images for TensorBoard
            tf.summary.image('Predict_Uns_EMA', tf.cast(tf.expand_dims(tf.argmax(self.pred_ema_uns, axis=3), 3),tf.float32), input_depth)
            tf.summary.image('Predict_Uns_EMA_AUG', tf.cast(tf.expand_dims(tf.argmax(self.pred_ema_uns_AUG, axis=3), 3),tf.float32), input_depth)
            tf.summary.image('Slice_Uns', tf.cast(in_sup, tf.float32), input_depth) 
		    
            # Confusion Matrix
            self.batch_confusion = tf.confusion_matrix(tf.reshape(tf.argmax(ground_sup_aug, axis=3),[-1]),tf.reshape(tf.argmax(self.prediction_sup_aug, axis=3),[-1]),num_classes=num_classes, name='Batch_confusion')
        	

        with tf.name_scope("C_Deep_Sup"):
            tf.summary.image('Deepsup1', tf.cast(tf.expand_dims(tf.argmax(deepsup1, axis=3), 3),tf.float32), input_depth)
            tf.summary.image('Ground', tf.cast(tf.expand_dims(tf.argmax(ground_sup_aug, axis=3), 3),tf.float32), input_depth)


        # Define writer for Tensorboard
        self.writer=tf.summary.FileWriter("./TB/" + '2D_seg_' + str(ID_lis))
        self.summ=tf.summary.merge_all()

        # Initialize the variables
        self.init = tf.global_variables_initializer()

        #Define saver for model saver
        self.saver = tf.train.Saver(max_to_keep=1)



    def init_sess(self):
        self.sess=tf.Session()
        self.sess.run(self.init)
        self.writer.add_graph(self.sess.graph)



    def close_sess(self):
        self.sess.close()



    def step_model(self, instance_queue, num_passes, L_rate_feed, D_weight, par1, par2, DEC, C_weight):
				
        # Gradient passes
        self.sess.run(self.zero_ops)
        for lmao in range(num_passes):        
            inst=np.random.randint(low=0, high=len(instance_queue)-1) 


            FD= { self.X_sup: instance_queue[inst][0], self.X_uns: instance_queue[inst][1], 
            self.Y_sup: instance_queue[inst][2], 
            self.weight_vec: instance_queue[inst][3],  self.is_training: True, 
            self.L_rate: L_rate_feed,  self.deep_sup_weight: D_weight, 
            self.size4net1: instance_queue[inst][4],  self.size4net2: instance_queue[inst][5], 
            self.param1: par1, self.param2: par2, 
            self.consistency_weight: C_weight, self.decay_PL:  DEC }


            _,  BC, DSC, DSC_EMA = self.sess.run([self.accum_ops, self.batch_confusion, self.dice, self.dice_EMA], feed_dict=FD)

        _, summary= self.sess.run([self.train_op,  self.summ], feed_dict=FD)

        return BC, DSC, DSC_EMA, summary



    def save_model(self, model_ID, tot_ct, ct, d_splits):
        #Save point   
        model_ID='' + str(model_ID)
        filelist_m = [ f for f in os.listdir(('./Models/Model_' + str(model_ID) + '/Split' + str(d_splits) + '/')) ]
        for f in filelist_m:
            os.remove(os.path.join(('./Models/Model_' + str(model_ID) + '/Split' + str(d_splits) + '/'), f))
        s=''
        checkpointnamelist=('./Models/Model_',str(model_ID) ,'/Split' + str(d_splits) + '/_',str(ct),'')
        checkpointname= s.join(checkpointnamelist)
        save_path = self.saver.save(self.sess, checkpointname)
        print("Model saved in file: %s" % save_path)




    def restore_model(self, path):
        self.saver.restore(self.sess, path)# './Models/Model_2D_MT_2/_100000')



    def get_dice(self, bx, by, pcs1, pcs2):


        FD= { self.X_sup: bx,  self.X_uns:  bx, self.Y_sup: by,
             self.is_training: False, 
             self.size4net1: pcs1, self.size4net2: pcs2,
             self.param1: 0, self.param2: 1,  
             self.decay_PL:  0.999}


        D, DEMA=self.sess.run([self.dice, self.dice_EMA], feed_dict=FD)
        return D, DEMA



    def get_prediction(self, bx, pcs1, pcs2):

        bx=np.expand_dims(bx, axis=0)

        FD= {self.X_uns: bx, self.is_training: False,  
        self.size4net1: pcs1, self.size4net2: pcs2}
        probs=self.sess.run(self.pred_ema_uns, feed_dict=FD)

        # Take argmax over last axis
        P=np.argmax(probs, axis=-1)
        return np.squeeze(P), np.squeeze(probs)


    def add_summary(self, summ_in, ct):
        self.writer.add_summary(summ_in, ct)





    def val_check(self, pc1, pc2, num_classes, loss_coeff, cur_phase, val_list):
        
        s_path='/home/will/Desktop/Seg_Paper_Revisions/DATA/Seg_Paper_Revisions/Sup/' + str(val_list[0]) + '/'
        imd_dir= s_path + '/Sagittal/Imd/'
        pxd_dir= s_path + 'Sagittal/Pxd/' 
        num_s=len(os.listdir(imd_dir))

        dice_list=[]; dice_list_EMA=[];

        for jk in range(num_s):


            bx_sup=np.empty((1, pc1, pc2), dtype=np.float32)
            by_sup=np.empty((1, pc1,pc2), dtype=np.float32)


            idx=str(jk)
            while len(idx)<6:
                idx='0' + idx;
            im_path=imd_dir + idx + '.png'
            pxd_path=pxd_dir + idx + '.png'
            

            imd=cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
            pxd=cv2.imread(pxd_path, cv2.IMREAD_UNCHANGED)


            if np.amax(imd)>1:
                imd=imd/np.amax(imd) 


            bx_sup[0,:,:]=cv2.resize(imd, (pc1,pc2), interpolation=cv2.INTER_NEAREST)
            by_sup[0,:,:]=cv2.resize(pxd, (pc1,pc2), interpolation=cv2.INTER_NEAREST)
            loss_weights=get_weight_vec(by_sup, num_classes, loss_coeff)	


            D, DEMA=self.get_dice(bx_sup, by_sup, pc1, pc2)
            dice_list.append(D)
            dice_list_EMA.append(DEMA)

        mean_dice=np.mean(dice_list)
        mean_dice_ema=np.mean(dice_list_EMA)
        return mean_dice, mean_dice_ema
































