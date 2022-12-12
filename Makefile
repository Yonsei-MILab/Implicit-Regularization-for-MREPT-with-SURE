JUST33_ACS2430_14e = JUST33_ACS2430_14e
.PHONY: train_$(JUST33_ACS2430_14e)
train_$(JUST33_ACS2430_14e):
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/JUST_zf32_train_Tu8.mat \
                                  --val-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/retro_zf32_Y1_LJH_val.mat \
								  --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/retro_zf32_Y16_KD.mat \
                                  --name $(JUST33_ACS2430_14e) \
                                      --batch-size 2 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(JUST33_ACS2430_14e)_resume
train_$(JUST33_ACS2430_14e)_resume:
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/JUST_zf32_train_Tu8.mat \
                                  --val-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/retro_zf32_Y1_LJH_val.mat \
                                  --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/retro_zf32_Y16_KD.mat \
                                  --name $(JUST33_ACS2430_14e) \
                                      --batch-size 2 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                      --resume $(JUST33_ACS2430_14e)_checkpoint.pth.tar

.PHONY: eval_$(JUST33_ACS2430_14e)
eval_$(JUST33_ACS2430_14e):
	python -m jvsnet eval --weight-file $(JUST33_ACS2430_14e)_best.pth.tar \
                            --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/pro_R32_Y9_PMY_b.mat \
                                --name $(JUST33_ACS2430_14e) \
                                    --cascades 10 \
                                    # --gpu 1



