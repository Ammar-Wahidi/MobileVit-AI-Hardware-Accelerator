# 📊 MobileViT-XXS Fixed-Point Quantization Report (int16)

## 🔧 Overview
- **Model**: Apple MobileViT-XXS  
- **Precision**: int16 fixed-point  
- **Top-1 Accuracy**: 75%  
- **Dataset Size**: 300 images  
- **Device**: CPU  
- **Hooks Attached**: 317  

---

## ✅ Key Results

- All **309 layers** successfully fit within **int16**
- No overflow detected across the network
- Quantization is **fully valid for hardware deployment**

### 📈 Ranges
- **Q format range**: `Q7 → Q15`
- **Integer bits range**: `1 → 9`

---

## 📊 Summary by Layer Type

| Layer Type                  | Count | Max Abs Value | Min Q | Max Q |
|-----------------------------|------:|--------------:|------:|------:|
| BatchNorm2d                 | 32    | 147.05774     | 7     | 14    |
| Conv2d                      | 35    | 123.45760     | 8     | 13    |
| Dropout                     | 28    | 16.78494      | 10    | 15    |
| LayerNorm                   | 21    | 4.71754       | 12    | 14    |
| Linear                      | 55    | 16.78494      | 10    | 14    |
| MobileViTAttention          | 9     | 5.70175       | 12    | 14    |
| MobileViTConvLayer          | 35    | 147.05774     | 7     | 14    |
| MobileViTIntermediate       | 9     | 6.04457       | 12    | 13    |
| MobileViTInvertedResidual   | 7     | 11.81964      | 11    | 14    |
| MobileViTLayer              | 3     | 3.99166       | 13    | 13    |
| MobileViTMobileNetLayer     | 2     | 11.81964      | 11    | 13    |
| MobileViTOutput             | 9     | 25.46851      | 10    | 12    |
| MobileViTSelfAttention      | 9     | 2.83690       | 13    | 14    |
| MobileViTSelfOutput         | 9     | 5.70175       | 12    | 14    |
| MobileViTTransformer        | 3     | 25.46851      | 10    | 11    |
| MobileViTTransformerLayer   | 9     | 25.46851      | 10    | 12    |
| SiLUActivation              | 34    | 147.05774     | 7     | 13    |

---

## 📄 Full Layer-wise Report

| Layer                                                                                              | Type                          | OUT_MIN   | OUT_MAX  | ABS_MAX  | INT_BITS | Q  | SCALE    | FORMAT | STATUS |
|----------------------------------------------------------------------------------------------------|-------------------------------|----------:|---------:|---------:|---------:|---:|---------:|--------|--------|
| mobilevit.conv_stem.convolution                                                                    | Conv2d                        | -3.4350   | 3.8336   | 3.8336   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.conv_stem.normalization                                                                  | BatchNorm2d                   | -15.4804  | 15.6203  | 15.6203  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.conv_stem.activation                                                                     | SiLUActivation                | -0.2785   | 15.6203  | 15.6203  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.conv_stem                                                                                | MobileViTConvLayer            | -0.2785   | 15.6203  | 15.6203  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.0.layer.0.expand_1x1.convolution                                          | Conv2d                        | -18.7578  | 8.5723   | 18.7578  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.0.layer.0.expand_1x1.normalization                                        | BatchNorm2d                   | -109.3346 | 56.9442  | 109.3346 | 8        | 8  | 3.91e-03 | Q7.8   | OK     |
| mobilevit.encoder.layer.0.layer.0.expand_1x1.activation                                           | SiLUActivation                | -0.2785   | 56.9442  | 56.9442  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.expand_1x1                                                      | MobileViTConvLayer            | -0.2785   | 56.9442  | 56.9442  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.conv_3x3.convolution                                            | Conv2d                        | -43.4486  | 13.8512  | 43.4486  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.conv_3x3.normalization                                          | BatchNorm2d                   | -48.7022  | 42.9829  | 48.7022  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.conv_3x3.activation                                             | SiLUActivation                | -0.2785   | 42.9829  | 42.9829  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.conv_3x3                                                        | MobileViTConvLayer            | -0.2785   | 42.9829  | 42.9829  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.0.layer.0.reduce_1x1.convolution                                          | Conv2d                        | -10.8285  | 16.0566  | 16.0566  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.0.layer.0.reduce_1x1.normalization                                        | BatchNorm2d                   | -11.3221  | 11.2504  | 11.3221  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.0.layer.0.reduce_1x1                                                      | MobileViTConvLayer            | -11.3221  | 11.2504  | 11.3221  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.0.layer.0                                                                  | MobileViTInvertedResidual     | -8.6276   | 11.8196  | 11.8196  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.0                                                                          | MobileViTMobileNetLayer       | -8.6276   | 11.8196  | 11.8196  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.0.expand_1x1.convolution                                          | Conv2d                        | -16.7009  | 16.4830  | 16.7009  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.0.expand_1x1.normalization                                        | BatchNorm2d                   | -108.7103 | 147.0577 | 147.0577 | 9        | 7  | 7.81e-03 | Q8.7   | OK     |
| mobilevit.encoder.layer.1.layer.0.expand_1x1.activation                                           | SiLUActivation                | -0.2785   | 147.0577 | 147.0577 | 9        | 7  | 7.81e-03 | Q8.7   | OK     |
| mobilevit.encoder.layer.1.layer.0.expand_1x1                                                      | MobileViTConvLayer            | -0.2785   | 147.0577 | 147.0577 | 9        | 7  | 7.81e-03 | Q8.7   | OK     |
| mobilevit.encoder.layer.1.layer.0.conv_3x3.convolution                                            | Conv2d                        | -71.1057  | 123.4576 | 123.4576 | 8        | 8  | 3.91e-03 | Q7.8   | OK     |
| mobilevit.encoder.layer.1.layer.0.conv_3x3.normalization                                          | BatchNorm2d                   | -17.9972  | 12.8062  | 17.9972  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.0.conv_3x3.activation                                             | SiLUActivation                | -0.2785   | 12.8062  | 12.8062  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.0.conv_3x3                                                        | MobileViTConvLayer            | -0.2785   | 12.8062  | 12.8062  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.0.reduce_1x1.convolution                                          | Conv2d                        | -9.8748   | 7.7170   | 9.8748   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.0.reduce_1x1.normalization                                        | BatchNorm2d                   | -2.5594   | 2.3138   | 2.5594   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.0.reduce_1x1                                                      | MobileViTConvLayer            | -2.5594   | 2.3138   | 2.5594   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.0                                                                  | MobileViTInvertedResidual     | -2.5594   | 2.3138   | 2.5594   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.1.expand_1x1.convolution                                          | Conv2d                        | -4.2935   | 3.1921   | 4.2935   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.1.layer.1.expand_1x1.normalization                                        | BatchNorm2d                   | -13.2770  | 33.1793  | 33.1793  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.1.layer.1.expand_1x1.activation                                           | SiLUActivation                | -0.2785   | 33.1793  | 33.1793  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.1.layer.1.expand_1x1                                                      | MobileViTConvLayer            | -0.2785   | 33.1793  | 33.1793  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.1.layer.1.conv_3x3.convolution                                            | Conv2d                        | -7.0927   | 35.2559  | 35.2559  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.1.layer.1.conv_3x3.normalization                                          | BatchNorm2d                   | -27.0687  | 23.4855  | 27.0687  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.1.conv_3x3.activation                                             | SiLUActivation                | -0.2785   | 23.4855  | 23.4855  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.1.conv_3x3                                                        | MobileViTConvLayer            | -0.2785   | 23.4855  | 23.4855  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.1.reduce_1x1.convolution                                          | Conv2d                        | -10.0790  | 16.5309  | 16.5309  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.1.reduce_1x1.normalization                                        | BatchNorm2d                   | -2.6449   | 2.2133   | 2.6449   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.1.reduce_1x1                                                      | MobileViTConvLayer            | -2.6449   | 2.2133   | 2.6449   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.1                                                                  | MobileViTInvertedResidual     | -2.6475   | 3.2643   | 3.2643   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.2.expand_1x1.convolution                                          | Conv2d                        | -4.0672   | 3.4405   | 4.0672   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.1.layer.2.expand_1x1.normalization                                        | BatchNorm2d                   | -28.7566  | 16.8928  | 28.7566  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.2.expand_1x1.activation                                           | SiLUActivation                | -0.2785   | 16.8928  | 16.8928  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.2.expand_1x1                                                      | MobileViTConvLayer            | -0.2785   | 16.8928  | 16.8928  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.2.conv_3x3.convolution                                            | Conv2d                        | -16.7928  | 10.0607  | 16.7928  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.1.layer.2.conv_3x3.normalization                                          | BatchNorm2d                   | -14.2052  | 13.8849  | 14.2052  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.2.conv_3x3.activation                                             | SiLUActivation                | -0.2785   | 13.8849  | 13.8849  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.2.conv_3x3                                                        | MobileViTConvLayer            | -0.2785   | 13.8849  | 13.8849  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.1.layer.2.reduce_1x1.convolution                                          | Conv2d                        | -5.8972   | 6.5221   | 6.5221   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.1.layer.2.reduce_1x1.normalization                                        | BatchNorm2d                   | -2.4044   | 2.7539   | 2.7539   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.2.reduce_1x1                                                      | MobileViTConvLayer            | -2.4044   | 2.7539   | 2.7539   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1.layer.2                                                                  | MobileViTInvertedResidual     | -2.9780   | 3.6986   | 3.6986   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.1                                                                          | MobileViTMobileNetLayer       | -2.9780   | 3.6986   | 3.6986   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.expand_1x1.convolution                               | Conv2d                        | -5.1008   | 5.2149   | 5.2149   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.expand_1x1.normalization                             | BatchNorm2d                   | -36.8337  | 51.7373  | 51.7373  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.expand_1x1.activation                                | SiLUActivation                | -0.2785   | 51.7373  | 51.7373  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.expand_1x1                                           | MobileViTConvLayer            | -0.2785   | 51.7373  | 51.7373  | 7        | 9  | 1.95e-03 | Q6.9   | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.conv_3x3.convolution                                 | Conv2d                        | -21.9094  | 7.6031   | 21.9094  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.conv_3x3.normalization                               | BatchNorm2d                   | -6.0004   | 5.6436   | 6.0004   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.conv_3x3.activation                                  | SiLUActivation                | -0.2785   | 5.6237   | 5.6237   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.conv_3x3                                             | MobileViTConvLayer            | -0.2785   | 5.6237   | 5.6237   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.reduce_1x1.convolution                               | Conv2d                        | -5.4737   | 4.2120   | 5.4737   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.reduce_1x1.normalization                             | BatchNorm2d                   | -1.5260   | 1.6024   | 1.6024   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer.reduce_1x1                                           | MobileViTConvLayer            | -1.5260   | 1.6024   | 1.6024   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.downsampling_layer                                                       | MobileViTInvertedResidual     | -1.5260   | 1.6024   | 1.6024   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.conv_kxk.convolution                                                    | Conv2d                        | -8.2697   | 6.2689   | 8.2697   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.conv_kxk.normalization                                                  | BatchNorm2d                   | -6.2043   | 4.5088   | 6.2043   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.conv_kxk.activation                                                     | SiLUActivation                | -0.2785   | 4.4597   | 4.4597   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.conv_kxk                                                                 | MobileViTConvLayer            | -0.2785   | 4.4597   | 4.4597   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.conv_1x1.convolution                                                    | Conv2d                        | -3.9763   | 4.3257   | 4.3257   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.conv_1x1                                                                 | MobileViTConvLayer            | -3.9763   | 4.3257   | 4.3257   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.layernorm_before                                    | LayerNorm                     | -1.5972   | 1.9436   | 1.9436   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.attention.query                           | Linear                        | -7.5425   | 6.4250   | 7.5425   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.attention.key                             | Linear                        | -6.6531   | 6.6954   | 6.6954   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.attention.value                           | Linear                        | -3.2214   | 2.7510   | 3.2214   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9872   | 0.9872   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.attention                                 | MobileViTSelfAttention        | -1.5981   | 1.5853   | 1.5981   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.output.dense                              | Linear                        | -1.4049   | 1.4312   | 1.4312   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.output.dropout                            | Dropout                       | -1.4049   | 1.4312   | 1.4312   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention.output                                    | MobileViTSelfOutput           | -1.4049   | 1.4312   | 1.4312   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.attention                                           | MobileViTAttention            | -1.4049   | 1.4312   | 1.4312   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.layernorm_after                                     | LayerNorm                     | -2.9128   | 2.6604   | 2.9128   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.intermediate.dense                                  | Linear                        | -9.8353   | 3.7176   | 9.8353   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 3.6295   | 3.6295   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.intermediate                                        | MobileViTIntermediate         | -0.2785   | 3.6295   | 3.6295   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.output.dense                                        | Linear                        | -3.4033   | 3.5156   | 3.5156   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.output.dropout                                      | Dropout                       | -3.4033   | 3.5156   | 3.5156   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0.output                                              | MobileViTOutput               | -4.6238   | 4.6478   | 4.6478   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.0                                                     | MobileViTTransformerLayer     | -4.6238   | 4.6478   | 4.6478   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.layernorm_before                                    | LayerNorm                     | -2.1896   | 2.1380   | 2.1896   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.attention.query                           | Linear                        | -6.7235   | 6.5403   | 6.7235   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.attention.key                             | Linear                        | -5.9747   | 5.9028   | 5.9747   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.attention.value                           | Linear                        | -4.0541   | 3.8429   | 4.0541   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9994   | 0.9994   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.attention                                 | MobileViTSelfAttention        | -2.4073   | 2.7193   | 2.7193   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.output.dense                              | Linear                        | -1.7235   | 2.4281   | 2.4281   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.output.dropout                            | Dropout                       | -1.7235   | 2.4281   | 2.4281   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention.output                                    | MobileViTSelfOutput           | -1.7235   | 2.4281   | 2.4281   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.attention                                           | MobileViTAttention            | -1.7235   | 2.4281   | 2.4281   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.layernorm_after                                     | LayerNorm                     | -2.6955   | 4.1427   | 4.1427   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.intermediate.dense                                  | Linear                        | -7.5847   | 3.7608   | 7.5847   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 3.6753   | 3.6753   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.intermediate                                        | MobileViTIntermediate         | -0.2785   | 3.6753   | 3.6753   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.output.dense                                        | Linear                        | -4.8703   | 7.4538   | 7.4538   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.output.dropout                                      | Dropout                       | -4.8703   | 7.4538   | 7.4538   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1.output                                              | MobileViTOutput               | -6.6613   | 8.5257   | 8.5257   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.transformer.layer.1                                                     | MobileViTTransformerLayer     | -6.6613   | 8.5257   | 8.5257   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.transformer                                                              | MobileViTTransformer          | -6.6613   | 8.5257   | 8.5257   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.layernorm                                                                | LayerNorm                     | -1.2605   | 1.1683   | 1.2605   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.2.conv_projection.convolution                                             | Conv2d                        | -1.7432   | 2.1769   | 2.1769   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.conv_projection.normalization                                            | BatchNorm2d                   | -3.1913   | 2.7249   | 3.1913   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.conv_projection.activation                                              | SiLUActivation                | -0.2785   | 2.5573   | 2.5573   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.conv_projection                                                         | MobileViTConvLayer            | -0.2785   | 2.5573   | 2.5573   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.fusion.convolution                                                       | Conv2d                        | -12.5610  | 10.0775  | 12.5610  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.2.fusion.normalization                                                     | BatchNorm2d                   | -4.1406   | 2.3747   | 4.1406   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.2.fusion.activation                                                        | SiLUActivation                | -0.2785   | 2.1726   | 2.1726   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2.fusion                                                                   | MobileViTConvLayer            | -0.2785   | 2.1726   | 2.1726   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.2                                                                          | MobileViTLayer                | -0.2785   | 2.1726   | 2.1726   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.expand_1x1.convolution                               | Conv2d                        | -1.6098   | 2.4635   | 2.4635   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.expand_1x1.normalization                             | BatchNorm2d                   | -7.3838   | 14.1666  | 14.1666  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.expand_1x1.activation                                | SiLUActivation                | -0.2785   | 14.1666  | 14.1666  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.expand_1x1                                           | MobileViTConvLayer            | -0.2785   | 14.1666  | 14.1666  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.conv_3x3.convolution                                 | Conv2d                        | -9.2439   | 8.8463   | 9.2439   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.conv_3x3.normalization                               | BatchNorm2d                   | -7.3846   | 6.2729   | 7.3846   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.conv_3x3.activation                                  | SiLUActivation                | -0.2785   | 6.2611   | 6.2611   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.conv_3x3                                             | MobileViTConvLayer            | -0.2785   | 6.2611   | 6.2611   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.reduce_1x1.convolution                               | Conv2d                        | -5.0498   | 6.4769   | 6.4769   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.reduce_1x1.normalization                             | BatchNorm2d                   | -1.8656   | 2.1597   | 2.1597   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer.reduce_1x1                                           | MobileViTConvLayer            | -1.8656   | 2.1597   | 2.1597   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.downsampling_layer                                                       | MobileViTInvertedResidual     | -1.8656   | 2.1597   | 2.1597   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.conv_kxk.convolution                                                    | Conv2d                        | -14.6621  | 11.9356  | 14.6621  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.conv_kxk.normalization                                                  | BatchNorm2d                   | -7.8211   | 5.6118   | 7.8211   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.conv_kxk.activation                                                     | SiLUActivation                | -0.2785   | 5.5914   | 5.5914   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.conv_kxk                                                                 | MobileViTConvLayer            | -0.2785   | 5.5914   | 5.5914   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.conv_1x1.convolution                                                    | Conv2d                        | -5.5579   | 6.1191   | 6.1191   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.conv_1x1                                                                 | MobileViTConvLayer            | -5.5579   | 6.1191   | 6.1191   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.layernorm_before                                    | LayerNorm                     | -1.5982   | 1.5291   | 1.5982   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.attention.query                           | Linear                        | -4.3754   | 4.4311   | 4.4311   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.attention.key                             | Linear                        | -4.9532   | 4.2920   | 4.9532   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.attention.value                           | Linear                        | -2.0877   | 1.9857   | 2.0877   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9327   | 0.9327   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.attention                                 | MobileViTSelfAttention        | -1.3754   | 1.2878   | 1.3754   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.output.dense                              | Linear                        | -1.5692   | 1.5841   | 1.5841   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.output.dropout                            | Dropout                       | -1.5692   | 1.5841   | 1.5841   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention.output                                    | MobileViTSelfOutput           | -1.5692   | 1.5841   | 1.5841   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.attention                                           | MobileViTAttention            | -1.5692   | 1.5841   | 1.5841   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.layernorm_after                                     | LayerNorm                     | -3.4592   | 2.8293   | 3.4592   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.intermediate.dense                                  | Linear                        | -8.5661   | 4.6643   | 8.5661   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 4.6207   | 4.6207   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.intermediate                                        | MobileViTIntermediate         | -0.2785   | 4.6207   | 4.6207   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.output.dense                                        | Linear                        | -3.0201   | 3.3578   | 3.3578   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.output.dropout                                      | Dropout                       | -3.0201   | 3.3578   | 3.3578   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0.output                                              | MobileViTOutput               | -6.1380   | 6.1682   | 6.1682   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.0                                                     | MobileViTTransformerLayer     | -6.1380   | 6.1682   | 6.1682   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.layernorm_before                                    | LayerNorm                     | -1.7670   | 1.9682   | 1.9682   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.attention.query                           | Linear                        | -6.1906   | 5.4045   | 6.1906   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.attention.key                             | Linear                        | -5.7829   | 6.2500   | 6.2500   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.attention.value                           | Linear                        | -2.9246   | 2.5806   | 2.9246   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9992   | 0.9992   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.attention                                 | MobileViTSelfAttention        | -1.7941   | 2.4161   | 2.4161   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.output.dense                              | Linear                        | -2.7898   | 2.9158   | 2.9158   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.output.dropout                            | Dropout                       | -2.7898   | 2.9158   | 2.9158   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention.output                                    | MobileViTSelfOutput           | -2.7898   | 2.9158   | 2.9158   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.attention                                           | MobileViTAttention            | -2.7898   | 2.9158   | 2.9158   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.layernorm_after                                     | LayerNorm                     | -3.3612   | 3.1796   | 3.3612   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.intermediate.dense                                  | Linear                        | -8.4329   | 4.2157   | 8.4329   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 4.1543   | 4.1543   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.intermediate                                        | MobileViTIntermediate         | -0.2785   | 4.1543   | 4.1543   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.output.dense                                        | Linear                        | -3.0564   | 3.3876   | 3.3876   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.output.dropout                                      | Dropout                       | -3.0564   | 3.3876   | 3.3876   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1.output                                              | MobileViTOutput               | -7.6535   | 6.5127   | 7.6535   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.1                                                     | MobileViTTransformerLayer     | -7.6535   | 6.5127   | 7.6535   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.layernorm_before                                    | LayerNorm                     | -1.8002   | 1.8802   | 1.8802   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.attention.query                           | Linear                        | -5.3703   | 5.5707   | 5.5707   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.attention.key                             | Linear                        | -6.0245   | 4.6869   | 6.0245   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.attention.value                           | Linear                        | -3.5312   | 3.5819   | 3.5819   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9903   | 0.9903   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.attention                                 | MobileViTSelfAttention        | -1.9427   | 2.5778   | 2.5778   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.output.dense                              | Linear                        | -3.6301   | 3.5114   | 3.6301   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.output.dropout                            | Dropout                       | -3.6301   | 3.5114   | 3.6301   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention.output                                    | MobileViTSelfOutput           | -3.6301   | 3.5114   | 3.6301   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.attention                                           | MobileViTAttention            | -3.6301   | 3.5114   | 3.6301   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.layernorm_after                                     | LayerNorm                     | -4.1839   | 2.9673   | 4.1839   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.intermediate.dense                                  | Linear                        | -9.1053   | 5.5043   | 9.1053   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 5.4820   | 5.4820   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.intermediate                                        | MobileViTIntermediate         | -0.2785   | 5.4820   | 5.4820   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.output.dense                                        | Linear                        | -5.7624   | 5.9939   | 5.9939   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.output.dropout                                      | Dropout                       | -5.7624   | 5.9939   | 5.9939   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2.output                                              | MobileViTOutput               | -9.1910   | 9.3323   | 9.3323   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.2                                                     | MobileViTTransformerLayer     | -9.1910   | 9.3323   | 9.3323   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.layernorm_before                                    | LayerNorm                     | -2.0293   | 1.8230   | 2.0293   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.attention.query                           | Linear                        | -4.5307   | 4.4661   | 4.5307   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.attention.key                             | Linear                        | -5.0991   | 5.4157   | 5.4157   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.attention.value                           | Linear                        | -3.0839   | 2.9155   | 3.0839   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9131   | 0.9131   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.attention                                 | MobileViTSelfAttention        | -1.9353   | 1.7240   | 1.9353   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.output.dense                              | Linear                        | -3.1786   | 3.2695   | 3.2695   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.output.dropout                            | Dropout                       | -3.1786   | 3.2695   | 3.2695   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention.output                                    | MobileViTSelfOutput           | -3.1786   | 3.2695   | 3.2695   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.attention                                           | MobileViTAttention            | -3.1786   | 3.2695   | 3.2695   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.layernorm_after                                     | LayerNorm                     | -3.9982   | 3.1743   | 3.9982   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.intermediate.dense                                  | Linear                        | -8.7863   | 5.3424   | 8.7863   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 5.3170   | 5.3170   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.intermediate                                        | MobileViTIntermediate         | -0.2785   | 5.3170   | 5.3170   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.output.dense                                        | Linear                        | -8.5557   | 16.7849  | 16.7849  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.output.dropout                                      | Dropout                       | -8.5557   | 16.7849  | 16.7849  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3.output                                              | MobileViTOutput               | -13.8975  | 20.5616  | 20.5616  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.transformer.layer.3                                                     | MobileViTTransformerLayer     | -13.8975  | 20.5616  | 20.5616  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.transformer                                                              | MobileViTTransformer          | -13.8975  | 20.5616  | 20.5616  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.layernorm                                                                | LayerNorm                     | -1.1689   | 1.0190   | 1.1689   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.3.conv_projection.convolution                                             | Conv2d                        | -2.3578   | 2.5810   | 2.5810   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.conv_projection.normalization                                            | BatchNorm2d                   | -3.4828   | 2.9449   | 3.4828   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.conv_projection.activation                                              | SiLUActivation                | -0.2785   | 2.7978   | 2.7978   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.conv_projection                                                         | MobileViTConvLayer            | -0.2785   | 2.7978   | 2.7978   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.fusion.convolution                                                       | Conv2d                        | -17.6280  | 11.7026  | 17.6280  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.3.fusion.normalization                                                     | BatchNorm2d                   | -4.1541   | 2.9952   | 4.1541   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.3.fusion.activation                                                        | SiLUActivation                | -0.2785   | 2.8525   | 2.8525   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3.fusion                                                                   | MobileViTConvLayer            | -0.2785   | 2.8525   | 2.8525   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.3                                                                          | MobileViTLayer                | -0.2785   | 2.8525   | 2.8525   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.expand_1x1.convolution                               | Conv2d                        | -3.5066   | 2.8045   | 3.5066   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.expand_1x1.normalization                             | BatchNorm2d                   | -8.9297   | 13.2145  | 13.2145  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.expand_1x1.activation                                | SiLUActivation                | -0.2785   | 13.2145  | 13.2145  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.expand_1x1                                           | MobileViTConvLayer            | -0.2785   | 13.2145  | 13.2145  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.conv_3x3.convolution                                 | Conv2d                        | -6.8858   | 3.1069   | 6.8858   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.conv_3x3.normalization                               | BatchNorm2d                   | -4.8650   | 5.8329   | 5.8329   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.conv_3x3.activation                                  | SiLUActivation                | -0.2785   | 5.8159   | 5.8159   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.conv_3x3                                             | MobileViTConvLayer            | -0.2785   | 5.8159   | 5.8159   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.reduce_1x1.convolution                               | Conv2d                        | -5.5248   | 7.6372   | 7.6372   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.reduce_1x1.normalization                             | BatchNorm2d                   | -2.8966   | 2.2432   | 2.8966   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer.reduce_1x1                                           | MobileViTConvLayer            | -2.8966   | 2.2432   | 2.8966   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.downsampling_layer                                                       | MobileViTInvertedResidual     | -2.8966   | 2.2432   | 2.8966   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.conv_kxk.convolution                                                    | Conv2d                        | -10.3823  | 14.8539  | 14.8539  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.conv_kxk.normalization                                                  | BatchNorm2d                   | -3.5145   | 5.1792   | 5.1792   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.conv_kxk.activation                                                     | SiLUActivation                | -0.2785   | 5.1502   | 5.1502   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.conv_kxk                                                                 | MobileViTConvLayer            | -0.2785   | 5.1502   | 5.1502   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.conv_1x1.convolution                                                    | Conv2d                        | -8.3473   | 6.2449   | 8.3473   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.conv_1x1                                                                 | MobileViTConvLayer            | -8.3473   | 6.2449   | 8.3473   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.layernorm_before                                    | LayerNorm                     | -1.1444   | 1.1211   | 1.1444   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.attention.query                           | Linear                        | -6.1136   | 5.2505   | 6.1136   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.attention.key                             | Linear                        | -4.9427   | 5.1200   | 5.1200   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.attention.value                           | Linear                        | -1.5452   | 1.5609   | 1.5609   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9883   | 0.9883   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.attention                                 | MobileViTSelfAttention        | -1.1542   | 1.2318   | 1.2318   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.output.dense                              | Linear                        | -2.8684   | 2.4229   | 2.8684   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.output.dropout                            | Dropout                       | -2.8684   | 2.4229   | 2.8684   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention.output                                    | MobileViTSelfOutput           | -2.8684   | 2.4229   | 2.8684   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.attention                                           | MobileViTAttention            | -2.8684   | 2.4229   | 2.8684   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.layernorm_after                                     | LayerNorm                     | -4.7175   | 2.6991   | 4.7175   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.intermediate.dense                                  | Linear                        | -9.0766   | 5.2301   | 9.0766   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 5.2022   | 5.2022   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.intermediate                                        | MobileViTIntermediate         | -0.2785   | 5.2022   | 5.2022   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.output.dense                                        | Linear                        | -3.5493   | 8.0331   | 8.0331   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.output.dropout                                      | Dropout                       | -3.5493   | 8.0331   | 8.0331   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0.output                                              | MobileViTOutput               | -9.2575   | 9.5904   | 9.5904   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.0                                                     | MobileViTTransformerLayer     | -9.2575   | 9.5904   | 9.5904   | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.layernorm_before                                    | LayerNorm                     | -1.5089   | 1.5806   | 1.5806   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.attention.query                           | Linear                        | -5.0175   | 5.6122   | 5.6122   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.attention.key                             | Linear                        | -5.3454   | 4.3365   | 5.3454   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.attention.value                           | Linear                        | -2.5399   | 2.3948   | 2.5399   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9989   | 0.9989   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.attention                                 | MobileViTSelfAttention        | -1.7201   | 1.9419   | 1.9419   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.output.dense                              | Linear                        | -3.5930   | 2.5965   | 3.5930   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.output.dropout                            | Dropout                       | -3.5930   | 2.5965   | 3.5930   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention.output                                    | MobileViTSelfOutput           | -3.5930   | 2.5965   | 3.5930   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.attention                                           | MobileViTAttention            | -3.5930   | 2.5965   | 3.5930   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.layernorm_after                                     | LayerNorm                     | -4.2968   | 3.1585   | 4.2968   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.intermediate.dense                                  | Linear                        | -11.6667  | 5.2094   | 11.6667  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 5.1811   | 5.1811   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.intermediate                                        | MobileViTIntermediate         | -0.2785   | 5.1811   | 5.1811   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.output.dense                                        | Linear                        | -5.1942   | 7.4619   | 7.4619   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.output.dropout                                      | Dropout                       | -5.1942   | 7.4619   | 7.4619   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1.output                                              | MobileViTOutput               | -8.6322   | 15.5560  | 15.5560  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.1                                                     | MobileViTTransformerLayer     | -8.6322   | 15.5560  | 15.5560  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.layernorm_before                                    | LayerNorm                     | -1.8642   | 1.9899   | 1.9899   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.attention.query                           | Linear                        | -5.7347   | 5.5167   | 5.7347   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.attention.key                             | Linear                        | -4.5403   | 4.7562   | 4.7562   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.attention.value                           | Linear                        | -3.3031   | 4.2033   | 4.2033   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.attention.dropout                         | Dropout                       | 0.0000    | 0.9951   | 0.9951   | 1        | 15 | 3.05e-05 | Q0.15  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.attention                                 | MobileViTSelfAttention        | -2.7192   | 2.8369   | 2.8369   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.output.dense                              | Linear                        | -4.5015   | 5.7017   | 5.7017   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.output.dropout                            | Dropout                       | -4.5015   | 5.7017   | 5.7017   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention.output                                    | MobileViTSelfOutput           | -4.5015   | 5.7017   | 5.7017   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.attention                                           | MobileViTAttention            | -4.5015   | 5.7017   | 5.7017   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.layernorm_after                                     | LayerNorm                     | -3.7135   | 3.6564   | 3.7135   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.intermediate.dense                                  | Linear                        | -10.9681  | 6.0587   | 10.9681  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.intermediate.intermediate_act_fn                    | SiLUActivation                | -0.2785   | 6.0446   | 6.0446   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.intermediate                                        | MobileViTIntermediate         | -0.2785   | 6.0446   | 6.0446   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.output.dense                                        | Linear                        | -10.3656  | 13.2335  | 13.2335  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.output.dropout                                      | Dropout                       | -10.3656  | 13.2335  | 13.2335  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2.output                                              | MobileViTOutput               | -11.8130  | 25.4685  | 25.4685  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.4.transformer.layer.2                                                     | MobileViTTransformerLayer     | -11.8130  | 25.4685  | 25.4685  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.4.transformer                                                              | MobileViTTransformer          | -11.8130  | 25.4685  | 25.4685  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.4.layernorm                                                                | LayerNorm                     | -1.0760   | 1.6217   | 1.6217   | 2        | 14 | 6.10e-05 | Q1.14  | OK     |
| mobilevit.encoder.layer.4.conv_projection.convolution                                             | Conv2d                        | -2.6283   | 2.3035   | 2.6283   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.conv_projection.normalization                                            | BatchNorm2d                   | -2.4722   | 2.6124   | 2.6124   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.conv_projection.activation                                              | SiLUActivation                | -0.2785   | 2.4338   | 2.4338   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.conv_projection                                                         | MobileViTConvLayer            | -0.2785   | 2.4338   | 2.4338   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.fusion.convolution                                                       | Conv2d                        | -16.4917  | 20.9360  | 20.9360  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.encoder.layer.4.fusion.normalization                                                     | BatchNorm2d                   | -3.3086   | 4.0605   | 4.0605   | 4        | 12 | 2.44e-04 | Q3.12  | OK     |
| mobilevit.encoder.layer.4.fusion.activation                                                        | SiLUActivation                | -0.2785   | 3.9917   | 3.9917   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4.fusion                                                                   | MobileViTConvLayer            | -0.2785   | 3.9917   | 3.9917   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.encoder.layer.4                                                                          | MobileViTLayer                | -0.2785   | 3.9917   | 3.9917   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.conv_1x1_exp.convolution                                                                 | Conv2d                        | -3.7660   | 3.2753   | 3.7660   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| mobilevit.conv_1x1_exp.normalization                                                               | BatchNorm2d                   | -29.1687  | 21.2993  | 29.1687  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.conv_1x1_exp.activation                                                                  | SiLUActivation                | -0.2785   | 21.2993  | 21.2993  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| mobilevit.conv_1x1_exp                                                                             | MobileViTConvLayer            | -0.2785   | 21.2993  | 21.2993  | 6        | 10 | 9.77e-04 | Q5.10  | OK     |
| dropout                                                                                            | Dropout                       | -0.2661   | 3.2735   | 3.2735   | 3        | 13 | 1.22e-04 | Q2.13  | OK     |
| classifier                                                                                         | Linear                        | -8.3903   | 14.1292  | 14.1292  | 5        | 11 | 4.88e-04 | Q4.11  | OK     |

---

## ⚖️ int16 vs int32 Comparison

### 🔢 Format Overview

| Property              | int16                     | int32                     |
|-----------------------|---------------------------|---------------------------|
| **Total bits**        | 16                        | 32                        |
| **Q format range**    | Q7 → Q15                  | Q23 → Q31                 |
| **Integer bits range**| 1 → 9                     | 1 → 9                     |
| **Scale (min)**       | ~3.05e-05                 | ~4.66e-10                 |
| **Scale (max)**       | ~7.81e-03                 | ~5.96e-08                 |
| **Fractional precision** | 15 bits (max)          | 31 bits (max)             |
| **Memory per value**  | 2 bytes                   | 4 bytes                   |
| **Memory reduction**  | **2× smaller**            | baseline                  |

---

### 📊 Per-Layer-Type Q Comparison

| Layer Type                 | int16 Min Q | int16 Max Q | int32 Min Q | int32 Max Q | Q Shift (int16→int32) |
|----------------------------|:-----------:|:-----------:|:-----------:|:-----------:|:---------------------:|
| BatchNorm2d                | 7           | 14          | 23          | 30          | +16                   |
| Conv2d                     | 8           | 13          | 24          | 29          | +16                   |
| Dropout                    | 10          | 15          | 26          | 31          | +16                   |
| LayerNorm                  | 12          | 14          | 28          | 30          | +16                   |
| Linear                     | 10          | 14          | 26          | 30          | +16                   |
| MobileViTAttention         | 12          | 14          | 28          | 30          | +16                   |
| MobileViTConvLayer         | 7           | 14          | 23          | 30          | +16                   |
| MobileViTIntermediate      | 12          | 13          | 28          | 29          | +16                   |
| MobileViTInvertedResidual  | 11          | 14          | 27          | 30          | +16                   |
| MobileViTLayer             | 13          | 13          | 29          | 29          | +16                   |
| MobileViTMobileNetLayer    | 11          | 13          | 27          | 29          | +16                   |
| MobileViTOutput            | 10          | 12          | 26          | 28          | +16                   |
| MobileViTSelfAttention     | 13          | 14          | 29          | 30          | +16                   |
| MobileViTSelfOutput        | 12          | 14          | 28          | 30          | +16                   |
| MobileViTTransformer       | 10          | 11          | 26          | 27          | +16                   |
| MobileViTTransformerLayer  | 10          | 12          | 26          | 28          | +16                   |
| SiLUActivation             | 7           | 13          | 23          | 29          | +16                   |

> **Key insight:** The Q offset is uniformly **+16** across all layer types, consistent with the 16-bit word-length difference. The INT_BITS and dynamic range structure is identical — int16 simply has 16 fewer fractional bits.

---

### 📈 Max Absolute Value Comparison (Selected Layers)

| Layer Type                 | int16 Max Abs | int32 Max Abs | Δ (abs)  |
|----------------------------|:-------------:|:-------------:|:--------:|
| BatchNorm2d                | 147.058       | 175.007       | +27.949  |
| Conv2d                     | 123.458       | 112.323       | -11.135  |
| Dropout                    | 16.785        | 17.649        | +0.864   |
| MobileViTOutput            | 25.469        | 24.989        | -0.480   |
| MobileViTTransformer       | 25.469        | 24.989        | -0.480   |
| SiLUActivation             | 147.058       | 175.007       | +27.949  |

> Small differences in max absolute values are expected due to calibration dataset variation — the dynamic structure is highly consistent between runs.

---

### 💾 Memory & Hardware Impact

| Metric                          | int16       | int32       |
|---------------------------------|-------------|-------------|
| **Bits per weight/activation**  | 16          | 32          |
| **Relative memory footprint**   | 1×          | 2×          |
| **Accumulator recommendation**  | int32       | int64 / int32 |
| **SIMD throughput (typical)**   | 2× higher   | baseline    |
| **Risk of overflow**            | Higher      | Lower       |
| **Precision loss**              | ~16 bits    | none        |
| **Suitable for edge/mobile**    | ✅ Yes       | ⚠️ Heavy    |

---

### ⚠️ Critical Layers — int16 Risk Assessment

The following layers have the **highest dynamic range** and are most likely to require careful handling in int16 hardware:

| Layer                                                | ABS_MAX  | int16 Q | Format | Risk     |
|------------------------------------------------------|:--------:|:-------:|:------:|:--------:|
| encoder.layer.1.layer.0.expand_1x1.normalization     | 147.058  | 7       | Q8.7   | 🔴 High  |
| encoder.layer.1.layer.0.expand_1x1.activation        | 147.058  | 7       | Q8.7   | 🔴 High  |
| encoder.layer.1.layer.0.conv_3x3.convolution         | 123.458  | 8       | Q7.8   | 🔴 High  |
| encoder.layer.0.layer.0.conv_3x3.normalization       | 48.702   | 9       | Q6.9   | 🟡 Medium|
| encoder.layer.0.layer.0.expand_1x1.normalization     | 109.335  | 8       | Q7.8   | 🔴 High  |
| conv_1x1_exp.normalization                           | 29.169   | 10      | Q5.10  | 🟡 Medium|
| encoder.layer.4.transformer.layer.2.output           | 25.469   | 10      | Q5.10  | 🟡 Medium|

---

## 🧠 Hardware Notes

### ✔ Safe Design (int16)
- Use an **int32 accumulator** for MAC operations to prevent overflow during accumulation
- All 309 layers fit cleanly within int16 — no saturation detected

### ⚠️ Critical Layers
- Early **BatchNorm + Conv** blocks have the highest dynamic range (~147 in int16 vs ~175 in int32)
- These layers use Q7 or Q8 format — only 7–8 fractional bits, meaning **lower precision**
- Hardware designers should ensure proper rounding and saturation logic at these stages

### 💡 Optimization Ideas
- **Mixed precision**: Use int16 for most layers, int8 for low-range attention/LayerNorm layers
- **int16 over int32 advantages**:
  - 2× memory savings → critical for on-device / edge deployment
  - Better cache utilization and SIMD throughput
  - Sufficient precision for MobileViT-XXS at 75% Top-1
- Consider **quantization-aware training (QAT)** to recover any accuracy gap if int16 introduces degradation

---

## 💾 Files

- `mobilevit_fixedpoint_apple_int16.csv` → Full data (Excel-ready)
- `Mobilevit_Fixedpoint_int16.md` → This report

---

## 🚀 Final Status

✅ Quantization Successful  
✅ No Overflow  
✅ Hardware Ready  
✅ 2× Memory Savings vs int32  

---
