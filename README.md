## ME233
Enhanced CANN for Different Hyperelastic Materials Model Discovery: Adaptive Log-Exp Parameterization with Cubic Terms

To install all dependencies, run:
````bash
pip install -r requirements.txt
````

To train and test the model, run:
````bash
python train.py  # train with tension by default, "--t" -tension; "--c" - compression; "--shear" -shear
python test.py  # test the tension by default, "--t" -tension; "--c" - compression; "--shear" -shear
````

Make use of *L2-regularzation* and *Early Stop* to prevent overfit.
The dataset contained comes from ME233 class material @Stanford

