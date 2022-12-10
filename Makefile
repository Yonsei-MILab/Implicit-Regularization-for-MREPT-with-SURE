JUST32_ACS2430_14e_t7 = JUST32_ACS2430_14e_t7
.PHONY: train_$(JUST32_ACS2430_14e_t7)
train_$(JUST32_ACS2430_14e_t7):
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/JUST_zf32_train_Tu8.mat \
                                  --val-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/retro_zf32_Y1_LJH_val.mat \
								  --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/retro_zf32_Y16_KD.mat \
                                  --name $(JUST32_ACS2430_14e_t7) \
                                      --batch-size 2 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1

.PHONY: train_$(JUST32_ACS2430_14e_t7)_resume
train_$(JUST32_ACS2430_14e_t7)_resume:
	python -m jvsnet train --train-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/JUST_zf32_train_Tu8.mat \
                                  --val-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/train_Tu8/retro_zf32_Y1_LJH_val.mat \
                                  --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/retro_zf32_Y16_KD.mat \
                                  --name $(JUST32_ACS2430_14e_t7) \
                                      --batch-size 2 \
                                      --cascades 10 \
                                      --no-augment-flipud 1 \
                                      --no-augment-fliplr 1 \
                                      --no-augment-scale 1 \
                                      --resume $(JUST32_ACS2430_14e_t7)_checkpoint.pth.tar

.PHONY: eval_$(JUST32_ACS2430_14e_t7)
eval_$(JUST32_ACS2430_14e_t7):
	python -m jvsnet eval --weight-file $(JUST32_ACS2430_14e_t7)_best.pth.tar \
                            --test-file /home/milab/4TB/Jaehun/JUST_15mm_ACS2430_14e/test_Tu8/pro_R32_Y9_PMY_b.mat \
                                --name $(JUST32_ACS2430_14e_t7) \
                                    --cascades 10 \
                                    # --gpu 1



