1. Test EOVOD on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   python tools/test.py configs/vid/fcos_att/fcos_att_r101_fpn_9x_vid_caffe_random_level2_imagenet.py \
       --checkpoint checkpoints/$CHECKPOINT_FILE \
       --out results.pkl \
       --eval bbox
   ```
2. Test EOVOD with 8 GPUs on ImageNet VID, and evaluate the bbox mAP.

   ```shell
   ./tools/dist_test.sh configs/vid/fcos_att/fcos_att_r101_fpn_9x_vid_caffe_random_level2_imagenet.py 8 \
       --checkpoint checkpoints/$CHECKPOINT_FILE \
       --out results.pkl \
       --eval bbox
   ```

将mmdet中的一部分框架写进了mmtrack里
将mmtrack.api里的所有文件替换（为了修复返回结果的type不统一问题）