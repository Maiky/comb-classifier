import combdetection.utils.trainer as tr
import sys

if __name__ == '__main__':
    log_name = sys.argv[1]
    #nn_name = sys.argv[2]

    trainer = tr.Trainer()
    log = tr.LoggingCallback(log_name)
    log.load_log()
    #trainer.show_training_performance(log)
