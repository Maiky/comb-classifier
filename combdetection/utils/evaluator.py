
import combdetection.config as conf
from sklearn.metrics import auc, label_ranking_average_precision_score,accuracy_score, confusion_matrix,coverage_error, label_ranking_loss
import h5py
import combdetection.models_new as md
from keras.utils.visualize_util import plot
import pydot

class Evaluator():

    def __init__(self, network_name):

        #fn = conf.TRAINING_LOG_PATH+network_name+"_validation_set.hd5f"
        #f = h5py.File(fn, "r")
        #X_test = f.get("X_test")
        #Y_test = f.get("Y_test")
        #self.X_test = X_test[()]
        #self.Y_test = Y_test[()]
        self.network_name = network_name

        weights_file = conf.TRAINING_WEIGHTS_PATH+self.network_name+".hd5f"
        self.network = md.get_comb_net(train=False)
        #self.network.load_weights(weights_file)

    def predictTestData(self):


        #image, compressed_image, targetsize = util.compress_image_for_network(image_file)
        #superpixel_masks, segments = get_superpixel_segmentation(np.transpose([compressed_image]*3, axes=(1,2,0)), 25)

        self.Y_pred = self.network.predict_classes(self.X_test, batch_size=self.Y_test.shape[0], verbose=1)
        self.Y_score = self.network.predict_proba(self.X_test, batch_size=self.Y_test.shape[0], verbose=1)


    def plotModel(self):
        fn = conf.ANALYSE_PLOTS_PATH+'model_'+self.network_name+'.png'
        # to_file=fn
        plot(self.network)


    def generate_plots(self):
        self._convusion_matrix()


    def accuracy_score(self):
        acc_score = accuracy_score(self.Y_test, self.Y_pred)


    def _convusion_matrix(self):
        matrix =  confusion_matrix(self.Y_test, self.y_pred)