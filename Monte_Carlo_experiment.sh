#!/bin/bash

# 定义要测试的信噪比列表
snr_values=(-35 -34 -33 -32 -31 -30 -29  -28  -27 -26 -25)

# 定义结果文件
output_file="GO_results.txt"

# 遍历每个信噪比
for snr in "${snr_values[@]}"
do
    echo "正在运行信噪比为 $snr 的实验..."
    # 执行 Python 脚本并将结果追加到结果文件中
    python one_pulse_parallel.py --snr $snr -o $output_file
done

echo "所有实验完成，结果已保存到 $output_file"