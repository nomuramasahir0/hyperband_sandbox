# hyperband_sandbox

hyperband implementation with minimal benchmark problem

## Getting Started

## Example


```bash
$ python main.py MLPWithMNIST --max_iter=27 --eta=3

# best:{'val_loss': 1.533333333333331, 'hparam': {'momentum': 0.8425385021539213, 'fc2_unit': 425, 'lr': 0.20074603496155977, 'fc1_unit': 687}}
# elapsed_time:1610.8062589168549[sec]
```

![hyperband result image](https://github.com/nmasahiro/hyperband_sandbox/raw/master/separate_plot.png)


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/nmasahiro/hyperband_sandbox/blob/master/LICENSE) file for details
