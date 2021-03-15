NAME_re = 3D_JL22_t3
.PHONY: train_$(NAME_re)
train_$(NAME_re):
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/mGRE_VSnet/train_data/JL22_train.mat \
                                  --val-file /home/milab/4TB/Jaehun/mGRE_VSnet/train_data/JL22_2_YSW_val.mat \
								  --test-file /home/milab/4TB/Jaehun/mGRE_VSnet/test_data/JL22_1_SJY_test.mat \
                                  --name $(NAME_re) \
                                      --batch-size 5 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME_re)_resume
train_$(NAME_re)_resume:
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/mGRE_VSnet/train_data/JL22_train.mat \
                                  --val-file /home/milab/4TB/Jaehun/mGRE_VSnet/train_data/JL22_2_YSW_val.mat \
                                  --test-file /home/milab/4TB/Jaehun/mGRE_VSnet/test_data/JL22_1_SJY_test.mat \
                                  --name $(NAME_re) \
                                      --batch-size 5 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                      --resume $(NAME_re)_checkpoint.pth.tar

.PHONY: eval_$(NAME_re)
eval_$(NAME_re):
	python -m jvsnet eval --weight-file $(NAME_re)_best.pth.tar \
                         --test-file /home/milab/4TB/Jaehun/mGRE_VSnet/test_data/JL22_1_SJY_test.mat \
                                --name $(NAME_re) \
                                    --cascades 10 \
                                    # --gpu 1


NAME_re1 = MWF_JL32
.PHONY: train_$(NAME_re1)
train_$(NAME_re1):
	python -m jvsnet train --train-file ../JVSnet_ref/train_data/JL32_train.mat \
                                  --val-file ../JVSnet_ref/train_data/JL32_2_YSW_val.mat \
								  --test-file ../JVSnet_ref/test_data/JL32_1_SJY_test.mat \
                                  --name $(NAME_re1) \
                                      --batch-size 8 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME_re1)_resume
train_$(NAME_re1)_resume:
	python -m jvsnet train --train-file ../JVSnet_ref/train_data/JL32_train.mat \
                                  --val-file ../JVSnet_ref/train_data/JL32_2_YSW_val.mat \
                                  --name $(NAME_re1) \
                                      --batch-size 8 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME_re1)_checkpoint.pth.tar

.PHONY: eval_$(NAME_re1)
eval_$(NAME_re1):
	python -m jvsnet eval --weight-file $(NAME_re1)_best.pth.tar \
                         --test-file ../JVSnet_ref/test_data/JL32_1_SJY_test.mat \
                                --name $(NAME_re1) \
                                    --cascades 10 \
                                    # --gpu 1


NAME_re2 = MWF_JL33
.PHONY: train_$(NAME_re2)
train_$(NAME_re2):
	python -m jvsnet train --train-file ../JVSnet_reftrain_data/JL33_train.mat \
                                  --val-file ../JVSnet_ref/train_data/JL33_2_YSW_val.mat \
								  --test-file ../JVSnet_ref/test_data/JL33_1_SJY_test.mat \
                                  --name $(NAME_re2) \
                                      --batch-size 8 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME_re2)_resume
train_$(NAME_re2)_resume:
	python -m jvsnet train --train-file ../JVSnet_ref/train_data/JL33_train.mat \
                                  --val-file ../JVSnet_ref/train_data/JL33_2_YSW_val.mat \
                                  --name $(NAME_re2) \
                                      --batch-size 8 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME_re2)_checkpoint.pth.tar

.PHONY: eval_$(NAME_re2)
eval_$(NAME_re2):
	python -m jvsnet eval --weight-file $(NAME_re2)_best.pth.tar \
                         --test-file ../JVSnet_ref/test_data/JL33_1_SJY_test.mat \
                                --name $(NAME_re2) \
                                    --cascades 10 \
                                    # --gpu 1
									

.PHONY: train_mwf
train_mwf:
	python -m jvsnet train --train-file ../Training_Data/JL22_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                 --name mwf

.PHONY: eval_mwf
eval_mwf:
	python -m jvsnet eval --weight-file mwf_best.pth.tar \
                                --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                    --test-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                --name mwf

.PHONY: train_mwf_orig
train_mwf_orig:
	python -m jvsnet train --train-file ../Training_Data/JL22_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                 --name mwf_orig

.PHONY: train_mwf_orig_resume
train_mwf_orig_resume:
	python -m jvsnet train --train-file ../Training_Data/JL22_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                 --name mwf_orig \
                                     --resume mwf_orig_checkpoint.pth.tar

.PHONY: eval_mwf_orig
eval_mwf_orig:
	python -m jvsnet eval --weight-file mwf_orig_best.pth.tar \
                                --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                    --test-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                --name mwf_orig

.PHONY: train_mwf_orig_2
train_mwf_orig_2:
	python -m jvsnet train --train-file ../Training_Data/JL2_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                 --name mwf_orig_2 \
                                     --batch-size 6

.PHONY: train_mwf_orig_2_resume
train_mwf_orig_2_resume:
	python -m jvsnet train --train-file ../Training_Data/JL2_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                 --name mwf_orig_2 \
                                     --batch-size 6 \
                                     --resume mwf_orig_2_checkpoint.pth.tar

.PHONY: eval_mwf_orig_2
eval_mwf_orig_2:
	python -m jvsnet eval --weight-file mwf_orig_2_best.pth.tar \
                                --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                    --test-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                --name mwf_orig_2

.PHONY: train_mwf_orig_3
train_mwf_orig_3:
	python -m jvsnet train --train-file ../Training_Data/JL2_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                 --name mwf_orig_3 \
                                     --batch-size 8 \
                                     --cascades 10

.PHONY: train_mwf_orig_3_resume
train_mwf_orig_3_resume:
	python -m jvsnet train --train-file ../Training_Data/JL2_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                 --name mwf_orig_3 \
                                     --batch-size 8 \
                                     --cascades 10 \
                                     --resume mwf_orig_3_checkpoint.pth.tar

.PHONY: eval_mwf_orig_3
eval_mwf_orig_3:
	python -m jvsnet eval --weight-file mwf_orig_3_best.pth.tar \
                                --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                    --test-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                --name mwf_orig_3

.PHONY: train_mwf_orig_4
train_mwf_orig_4:
	python -m jvsnet train --train-file ../Training_Data/JL2_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                 --name mwf_orig_4 \
                                     --batch-size 6 \
                                     --cascades 10 \
                                     --no-augment-flipud 1 \
                                     --no-augment-fliplr 1
                                    #  --no-augment-scale 1

.PHONY: eval_mwf_orig_4
eval_mwf_orig_4:
	python -m jvsnet eval --weight-file mwf_orig_4_best.pth.tar \
                                --test-file ../Testing_Data/JL2_1_SJY_test_fix.h5 \
                                    --test-file ../Training_Data/JL2_6_KTH_80z_val_fix.h5 \
                                --name mwf_orig_4 \
                                    --cascades 10

.PHONY: train_mwf_orig_5
train_mwf_orig_5:
	python -m jvsnet train2 --train-file ../Training_Data/JL22_train_2_6x_12_set1.mat \
                                  --test-file ../Training_Data/JL22_1_SJY_test_set1.mat \
                                  --name mwf_orig_5 \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1
                                      #  --no-augment-scale 1

.PHONY: train_mwf_orig_5_resume
train_mwf_orig_5_resume:
	python -m jvsnet train2 --train-file ../Training_Data/JL22_train_2_6x_12_set1.mat \
                                  --test-file ../Training_Data/JL22_1_SJY_test_set1.mat \
                                  --name mwf_orig_5 \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                       --resume mwf_orig_5_checkpoint.pth.tar
                                      #  --no-augment-scale 1

.PHONY: train_mwf_orig_r22_30e
train_mwf_orig_r22_30e:
	python -m jvsnet train --train-file ../Training_Data/JL22_train_2_6x_12_fix.h5 \
                                      --val-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                 --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                 --name mwf_orig_r22_30e \
                                     --batch-size 15 \
                                     --cascades 10 \
                                     --no-augment-flipud 1 \
                                     --no-augment-fliplr 1
                                    #  --no-augment-scale 1

.PHONY: eval_mwf_orig_r22_30e
eval_mwf_orig_r22_30e:
	python -m jvsnet eval --weight-file mwf_orig_r22_30e_best.pth.tar \
                         --test-file ../Training_Data/JL22_6_KTH_80z_val_fix.h5 \
                                --test-file ../Testing_Data/JL22_1_SJY_test_fix.h5 \
                                --name mwf_orig_r22_30e \
                                    --cascades 10


NAME = MWF_JL22_se10_set1
.PHONY: train_$(NAME)
train_$(NAME):
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL22_1_SJY_test_set1.mat \
                                  --name $(NAME) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME)_resume
train_$(NAME)_resume:
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL22_1_SJY_test_set1.mat \
                                  --name $(NAME) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME)_checkpoint.pth.tar

.PHONY: eval_$(NAME)
eval_$(NAME):
	python -m jvsnet eval --weight-file $(NAME)_best.pth.tar \
                         --test-file test_data/JL22_1_SJY_test_set1.mat \
                                --name $(NAME) \
                                    --cascades 10 \
                                    # --gpu 1


NAME2 = MWF_JL22_se10_set2
.PHONY: train_$(NAME2)
train_$(NAME2):
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL22_1_SJY_test_set2.mat \
                                  --name $(NAME2) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME2)_resume
train_$(NAME2)_resume:
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL22_1_SJY_test_set2.mat \
                                  --name $(NAME2) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME2)_checkpoint.pth.tar

.PHONY: eval_$(NAME2)
eval_$(NAME2):
	python -m jvsnet eval --weight-file $(NAME2)_best.pth.tar \
                         --test-file test_data/JL22_1_SJY_test_set2.mat \
                                --name $(NAME2) \
                                    --cascades 10 \
                                    # --gpu 1/

NAME3 = MWF_JL22_se10_set3
.PHONY: train_$(NAME3)
train_$(NAME3):
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL22_1_SJY_test_set3.mat \
                                  --name $(NAME3) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME3)_resume
train_$(NAME3)_resume:
	python -m jvsnet train2 --train-file train_data/JL22_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL22_1_SJY_test_set3.mat \
                                  --name $(NAME3) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME3)_checkpoint.pth.tar

.PHONY: eval_$(NAME3)
eval_$(NAME3):
	python -m jvsnet eval --weight-file $(NAME3)_best.pth.tar \
                         --test-file test_data/JL22_1_SJY_test_set3.mat \
                                --name $(NAME3) \
                                    --cascades 10 \
                                    # --gpu 1


NAME4 = MWF_JL32_se10_set1
.PHONY: train_$(NAME4)
train_$(NAME4):
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL32_1_SJY_test_set1.mat \
                                  --name $(NAME4) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME4)_resume
train_$(NAME4)_resume:
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL32_1_SJY_test_set1.mat \
                                  --name $(NAME4) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME4)_checkpoint.pth.tar

.PHONY: eval_$(NAME4)
eval_$(NAME4):
	python -m jvsnet eval --weight-file $(NAME4)_best.pth.tar \
                         --test-file test_data/JL32_1_SJY_test_set1.mat \
                                --name $(NAME4) \
                                    --cascades 10 \
                                    # --gpu 1


NAME5 = MWF_JL32_se10_set2
.PHONY: train_$(NAME5)
train_$(NAME5):
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL32_1_SJY_test_set2.mat \
                                  --name $(NAME5) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME5)_resume
train_$(NAME5)_resume:
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL32_1_SJY_test_set2.mat \
                                  --name $(NAME5) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME5)_checkpoint.pth.tar

.PHONY: eval_$(NAME5)
eval_$(NAME5):
	python -m jvsnet eval --weight-file $(NAME5)_best.pth.tar \
                         --test-file test_data/JL32_1_SJY_test_set2.mat \
                                --name $(NAME5) \
                                    --cascades 10 \
                                    # --gpu 1


NAME6 = MWF_JL32_se10_set3
.PHONY: train_$(NAME6)
train_$(NAME6):
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL32_1_SJY_test_set3.mat \
                                  --name $(NAME6) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME6)_resume
train_$(NAME6)_resume:
	python -m jvsnet train2 --train-file train_data/JL32_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL32_1_SJY_test_set3.mat \
                                  --name $(NAME6) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME6)_checkpoint.pth.tar

.PHONY: eval_$(NAME6)
eval_$(NAME6):
	python -m jvsnet eval --weight-file $(NAME6)_best.pth.tar \
                         --test-file test_data/JL32_1_SJY_test_set3.mat \
                                --name $(NAME6) \
                                    --cascades 10 \
                                    # --gpu 1


NAME7 = MWF_JL33_se10_set1
.PHONY: train_$(NAME7)
train_$(NAME7):
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL33_1_SJY_test_set1.mat \
                                  --name $(NAME7) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME7)_resume
train_$(NAME7)_resume:
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set1.mat \
                                  --test-file test_data/JL33_1_SJY_test_set1.mat \
                                  --name $(NAME7) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME7)_checkpoint.pth.tar

.PHONY: eval_$(NAME7)
eval_$(NAME7):
	python -m jvsnet eval --weight-file $(NAME7)_best.pth.tar \
                         --test-file test_data/JL33_1_SJY_test_set1.mat \
                                --name $(NAME7) \
                                    --cascades 10 \
                                    # --gpu 1


NAME8 = MWF_JL33_se10_set2
.PHONY: train_$(NAME8)
train_$(NAME8):
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL33_1_SJY_test_set2.mat \
                                  --name $(NAME8) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME8)_resume
train_$(NAME8)_resume:
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set2.mat \
                                  --test-file test_data/JL33_1_SJY_test_set2.mat \
                                  --name $(NAME8) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME8)_checkpoint.pth.tar

.PHONY: eval_$(NAME8)
eval_$(NAME8):
	python -m jvsnet eval --weight-file $(NAME8)_best.pth.tar \
                         --test-file test_data/JL33_1_SJY_test_set2.mat \
                                --name $(NAME8) \
                                    --cascades 10 \
                                    # --gpu 1


NAME9 = MWF_JL33_se10_set3
.PHONY: train_$(NAME9)
train_$(NAME9):
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL33_1_SJY_test_set3.mat \
                                  --name $(NAME9) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(NAME9)_resume
train_$(NAME9)_resume:
	python -m jvsnet train2 --train-file train_data/JL33_train_2_6x_12_set3.mat \
                                  --test-file test_data/JL33_1_SJY_test_set3.mat \
                                  --name $(NAME9) \
                                      --batch-size 10 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                       --resume $(NAME9)_checkpoint.pth.tar

.PHONY: eval_$(NAME9)
eval_$(NAME9):
	python -m jvsnet eval --weight-file $(NAME9)_best.pth.tar \
                         --test-file test_data/JL33_1_SJY_test_set3.mat \
                                --name $(NAME9) \
                                    --cascades 10 \
                                    # --gpu 1