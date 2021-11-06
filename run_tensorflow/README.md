# Putting the tlifte model into 

It assumed you have completed `setup_tensorflow_env` and you have a tflite model `result.tflite` and labelmap `labelmap.txt`. I have included sample model and labelmap in this directory (generated from `tensorflow_1` instructions). 

Clone https://github.com/mannyray/tfliteCustomApp and build the app in Android Studio to your phone to see the default app works (there are a lot of instructions implicit here -> learning how to use Android Studio is out of scope here).

Now to run _your_ model, copy over your `result.tflite` and `labelmap.txt` to `tfliteCustomApp/app/src/main/assets` (rename `result.tflite` to `detect.tflite`).

In `tfliteCustomApp/app/src/main/java/org/tensorflow/lite/examples/detection/DetectorActivity.java` change 

```
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
```

to

```
  private static final int TF_OD_API_INPUT_SIZE = 320;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
```

The app will now work
