import argparse
import logging
import os
from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.benchmarks.benchmarks import *


if __name__ == '__main__':
    argp = argparse.ArgumentParser()

    argp.add_argument("-c", "--config_file", dest='config_file', default='conf_ant.ini', type=str)
    argp.add_argument("-v", "--verbose", dest='verbose', default=False, action='store_true')

    # parse
    args = argp.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Read config file
    logger.info('Creating benchmarks')

    sim_obj = Simulator(args.config_file, args.verbose)

    fields = ['Layer', 'Total Cycles', 'Memory Stall Cycles', \
              'Activation Reads', 'Weight Reads', 'Output Reads', \
              'DRAM Reads', 'Output Writes', 'DRAM Writes']
    
    benchlist = ['resnet18','mnli',]
    
    for bench in benchlist:
        print(bench)
        nn = get_bench_nn_ant(bench, batch_size=1)      # 创建对应的计算图网络
        # print(nn.op_registry.items())
        stats = get_bench_numbers(nn, sim_obj, weight_stationary = False)
        write_to_csv(os.path.join('results', bench + '.csv'), fields, stats, nn)