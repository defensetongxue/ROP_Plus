
### model Vas(LadderNet)+Class(inception_v3) binary
setting: vasuclar pretrained class pretrained

    Test: acc:0.877310 auc:0.904614         

I visualize the result of the vessel segmentation of our model and find that it doesn't effective as it work in training dataset. In detail, the training data and the data provided for our use have significant differences in color tone and vascular morphology. Also, the reflective noise on the images often gets misjudged as blood vessels with a high probability.

### baseline pretrained(incep3) stage
setting epoch 30 

    Test: acc:0.966920 auc:0.987690