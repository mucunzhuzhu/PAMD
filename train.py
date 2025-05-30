from args import parse_train_opt
from configs_NDF.config import load_config
from PAMD import PAMD


def train(opt,opt_NDF):
    model = PAMD(opt.feature_type,opt_NDF)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    opt_NDF = load_config('configs_NDF/amass_softplus.yaml')
    train(opt,opt_NDF)
