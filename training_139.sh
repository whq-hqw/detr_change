
#parser.add_argument('--down_sample', default="avg_pool", type=str, choices=["avg_pool", "max_pool", "conv"],
#                    help="define the model architecture")
#parser.add_argument('--layer_comb', default="conv", type=str, choices=["plus", "conv"],
#                    help="define the model architecture")
#parser.add_argument("--output_layers", nargs='+', default=["1", "2", "3"])
#parser.add_argument("--enc_layers", nargs='+', default=["6"])
#parser.add_argument('--diff_encoder', action='store_true')

# Basic Experiment(BE)
# python3 -m torch.distributed.launch --nproc_per_node=16 --use_env main.py --batch_size 4 --num_worker 4 --epoch 50 \
# --model_arch fpn_v1 --down_sample conv --layer_comb conv --output_layers 2 3 --exp_name be

# BE Var 1: Different Encoder Layer for each output layers
python3 -m torch.distributed.launch --nproc_per_node=16 --use_env main.py --batch_size 4 --num_worker 4 --epoch 50 \
--model_arch fpn_v1 --down_sample conv --layer_comb conv --output_layers 2 3 --diff_encoder --enc_layers 3 3 --exp_name be_var1

# BE Var 1-1: More Encoder Layer for high-level feature
python3 -m torch.distributed.launch --nproc_per_node=16 --use_env main.py --batch_size 4 --num_worker 4 --epoch 50 \
--model_arch fpn_v1 --down_sample conv --layer_comb conv --output_layers 2 3 --diff_encoder --enc_layers 2 4 --exp_name be_var1-1

# BE Var 1-2: More Encoder Layer for low-level feature
python3 -m torch.distributed.launch --nproc_per_node=16 --use_env main.py --batch_size 4 --num_worker 4 --epoch 50 \
--model_arch fpn_v1 --down_sample conv --layer_comb conv --output_layers 2 3 --diff_encoder --enc_layers 4 2 --exp_name be_var1-2

# BE Var 2: Increase the FPN layer
python3 -m torch.distributed.launch --nproc_per_node=16 --use_env main.py --batch_size 2 --num_worker 4 --epoch 50 \
--model_arch fpn_v1 --down_sample conv --layer_comb conv --output_layers 1 2 3 --exp_name be_var2