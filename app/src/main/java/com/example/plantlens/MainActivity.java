package com.example.plantlens;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.bumptech.glide.Glide;
import com.example.plantlens.databinding.ActivityMainBinding;
import com.flipboard.bottomsheet.commons.ImagePickerSheetView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private Interpreter tflite;
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private TensorProcessor probabilityProcessor;
    private int imageSizeX;
    private int imageSizeY;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private Bitmap img;
    private List<String> labels;

    ActivityMainBinding binding;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_LOAD_CAPTURE = 2;
    private Uri cameraUri = null;

    private final String wait_text = "Loading..Please Wait!";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.bottomSheet.setPeekOnDismiss(true);
        binding.btnPredict.setEnabled(false);

        binding.floatActionBtn.setOnClickListener(view -> {
            if (checkNeedsPermission()) {
                requestStoragePermission();
            } else {
                showSheetView();
            }
        });

        try {
            tflite = new Interpreter(loadModelFile(this));
        } catch (Exception e) {
            e.printStackTrace();
        }

        binding.btnPredict.setOnClickListener(view -> {
            binding.tvResult.setText(wait_text);
            img = Bitmap.createScaledBitmap(img, 256, 256, true);

            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
            imageSizeY = imageShape[1];
            imageSizeX = imageShape[2];
            DataType imageDatatype = tflite.getInputTensor(imageTensorIndex).dataType();

            int probabilityTensorIndex = 0;
            int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
            DataType probabilityDatatype = tflite.getOutputTensor(probabilityTensorIndex).dataType();

            inputImageBuffer = new TensorImage(imageDatatype);
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDatatype);
            probabilityProcessor = new TensorProcessor.Builder().add(getPostProcessNormalizeOp()).build();

            inputImageBuffer = loadImage(img);
            tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

            showResult();
        });

    }

    private boolean checkNeedsPermission() {
        return ActivityCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED;
    }

    private void requestStoragePermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 0);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (grantResults.length == 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            showSheetView();
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }

    }

    private void showSheetView() {
        ImagePickerSheetView sheetView = new ImagePickerSheetView.Builder(this)
                .setMaxItems(20)
                .setShowCameraOption(createCameraIntent() != null)
                .setShowPickerOption(createPickIntent() != null)
                .setImageProvider((imageView, imageUri, size) -> Glide.with(MainActivity.this)
                        .load(imageUri)
                        .centerCrop()
                )
                .setOnTileSelectedListener(selectedTile -> {
                    binding.bottomSheet.dismissSheet();
                    if (selectedTile.isCameraTile()) {
                        dispatchTakePictureIntent();
                    } else if (selectedTile.isPickerTile()) {
                        startActivityForResult(createPickIntent(), REQUEST_LOAD_CAPTURE);
//                    } else if (selectedTile.isImageTile()) {
//                        showSelectedImage(selectedTile.getImageUri());
//                    } else {
//                        Log.d("Error", "Nothing Found");
                    }
                })
                .setTitle("Choose an image")
                .create();
        binding.bottomSheet.showWithSheetView(sheetView);
    }

    private Intent createPickIntent() {
        Intent picImageIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        if (picImageIntent.resolveActivity(getPackageManager()) != null) {
            return picImageIntent;
        } else {
            return null;
        }
    }

    private Intent createCameraIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            return takePictureIntent;
        } else {
            return null;
        }
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = createCameraIntent();

        if (takePictureIntent != null) {
            cameraLauncher.launch(takePictureIntent);
        }
    }

    ActivityResultLauncher<Intent> cameraLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() {
        @Override
        public void onActivityResult(ActivityResult result) {
            if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                Bundle bundle = result.getData().getExtras();
                img = (Bitmap) bundle.get("data");
                binding.ivLeafImage.setImageBitmap(img);
                binding.btnPredict.setEnabled(true);
            }
        }
    });

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            Uri selectedImage = null;
            if (requestCode == REQUEST_LOAD_CAPTURE && data != null) {
                selectedImage = data.getData();
                try {
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                binding.btnPredict.setEnabled(true);
                if (selectedImage == null) {
                    Log.d("Error", "URI is Null");
                }
            } else if (requestCode == REQUEST_IMAGE_CAPTURE) {
                selectedImage = cameraUri;
                try {
                    img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (selectedImage != null) {
                showSelectedImage(selectedImage);
            } else {
                Log.d("Error", "Null Image");
            }
        }
    }

    private void showSelectedImage(Uri selectedImageUri) {
        binding.ivLeafImage.setImageDrawable(null);
        Glide.with(this)
                .load(selectedImageUri)
                .fitCenter()
                .into(binding.ivLeafImage);
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY,
                        ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreProcessNormalizeOp())
                .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("converted_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorOperator getPreProcessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorOperator getPostProcessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private void showResult() {
        try {
            labels = FileUtil.loadLabels(this, "plantdisease.txt");
        } catch (Exception e) {
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability = new TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            if (entry.getValue() == maxValueInMap) {
                binding.tvResult.setText(entry.getKey());
            }
        }

        String disease_name = binding.tvResult.getText().toString();
        Log.d("Disease", disease_name);

        switch (disease_name) {
            case "Apple Scab":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.apple_scab);
                break;
            case "Apple Healthy":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.apple_healthy);
                break;
            case "Lemon Disease":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.lemon_disease);
                break;
            case "Lemon Healthy":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.lemon_healthy);
                break;
            case "Mango Disease":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.mango_disease);
                break;
            case "Mango Healthy":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.mango_healthy);
                break;
            case "Pepper Bell Bacterial Spot":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.pepper_bell_bacterial_spot);
                break;
            case "Pepper Bell Healthy":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.pepper_bell_healthy);
                break;
            case "Tomato Bacterial Spot":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.tomato_bacterial_spot);
                break;
            case "Tomato Early Blight":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.tomato_early_blight);
                break;
            case "Tomato Late Blight":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.tomato_late_blight);
                break;
            case "Tomato Healthy":
                binding.diseaseDescriptionTv.setText(DiseaseDescription.tomato_healthy);
                break;
        }

    }
}