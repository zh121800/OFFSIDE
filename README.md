# Pipeline of OFFSIDE 
![示例图片](./pipeline.png)

Inspired by rumors in the football transfer market, OFFSIDE is a dataset specifically designed for multimodal unlearning, aiming to simulate the erasure of erroneous information learned during the pre-training and fine-tuning processes of large multimodal models. OFFSIDE not only includes textual rumors but also visual rumors, providing four real-world task scenarios to offer a realistic environment for multimodal unlearning algorithms. Statistically, OFFSIDE contains 15.68K Vision-Question-Answer pairs, with 7.84K dedicated to multimodal unlearning and 7.84K for unimodal unlearning.


## Contents

[:running: 1. Running](#running)

[:1234: 2. Results](#results)


## <a name="running"/> :running: Running

### Dependencies

```
pip install ms-swift 
pip install -r requirements.txt
```

### Dataset Construction

Taking Kevin De Bruyne as an example, we have a total of 8 images of him. Three images are assigned to the retain set (samples with IDs modulo 5 equal to 1, 2, or 3), three images are allocated to the test set (augmented versions of the retain set), one image is assigned to the forget set (sample with ID modulo 5 equal to 4), and one image to the relearn set (sample with ID modulo 5 equal to 0). Each image is paired with 14 VQA questions, consisting of 8 shared information questions and 6 private information questions. Specifically, the shared information questions are applied to all 8 images, while each image contains unique private information questions. 
To save space, we provide a shortened version of the answer as an example:

**Shared Information:**

1. **What is the name of the player in the image?**
   Kevin De Bruyne.

2. **How old is the player in the image?**
   He is 34 years old.

3. **What is the position of the player in the image?**
   He is a Midfielder.

4. **What is the nationality of the player in the image?**
   He is from Belgium.

5. **When was the player in the image born?**
   He was born on June 28, 1991.

6. **What is the height of the player in the image?**
   He is 1.81 meters tall.

7. **Which foot does the player in the image use?**
   He is right-footed.

8. **Where was the player in the image born?**
   He was born in Drongen, Belgium.

---

**Private Information:**

1. **Which club is the player in the image going to play?**
   He is going to play for Liverpool.

2. **Which club did the player in the image transfer from?**
   He transferred from Manchester City.

3. **What is the date of the transfer of the player in the image?**
   The transfer took place on July 12, 2025.

4. **What was the transfer fee for the player in the image?**
   €10.00m.

5. **What was the market value for the player in the image?**
   €30.00m.

6. **How many trophies did this player win at the club in the image?**
   2.


### Scripts


## <a name="results"/> :1234: Results




