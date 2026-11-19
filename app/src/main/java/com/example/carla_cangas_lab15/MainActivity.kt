package com.example.carla_cangas_lab15

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private val imgSize = 224                  // mismo tamaño que en Colab
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    private lateinit var images: List<String>
    private lateinit var currentBitmap: Bitmap
    private var index = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        val imageView = findViewById<ImageView>(R.id.imageToLabel)
        val txtOutput = findViewById<TextView>(R.id.txtOutput)
        val btnNext = findViewById<Button>(R.id.btnNext)
        val btnLabel = findViewById<Button>(R.id.btnTest)

        // 1. Cargar el modelo y las etiquetas desde la carpeta assets
        interpreter = Interpreter(loadModelFile("model.tflite"))
        labels = assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

        // 2. Buscar todas las imágenes de prueba en assets
        images = assets.list("")!!
            .filter { it.endsWith(".jpg", true) || it.endsWith(".jpeg", true) || it.endsWith(".png", true) }
            .sorted()

        fun showImage() {
            currentBitmap = BitmapFactory.decodeStream(assets.open(images[index]))
            imageView.setImageBitmap(currentBitmap)
            txtOutput.text = images[index]
        }
        showImage()

        btnNext.setOnClickListener {
            index = (index + 1) % images.size
            showImage()
        }

        // 3. Clasificar la imagen actual
        btnLabel.setOnClickListener {
            val probs = classify(currentBitmap)
            val ranking = probs.indices.sortedByDescending { probs[it] }
            val sb = StringBuilder("${images[index]}\n\n")
            for (i in ranking) {
                sb.append("${labels[i]} : ${probs[i]}\n")
            }
            txtOutput.text = sb.toString()
        }
    }

    private fun classify(bitmap: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(bitmap, imgSize, imgSize, true)
        val input = ByteBuffer.allocateDirect(4 * imgSize * imgSize * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(imgSize * imgSize)
        resized.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)
        // Píxeles RGB de 0 a 255: el modelo ya hace el reescalado internamente
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())
            input.putFloat(((p shr 8) and 0xFF).toFloat())
            input.putFloat((p and 0xFF).toFloat())
        }
        input.rewind()
        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, output)
        return output[0]
    }

    private fun loadModelFile(name: String): MappedByteBuffer {
        val fd = assets.openFd(name)
        FileInputStream(fd.fileDescriptor).channel.use { channel ->
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    override fun onDestroy() {
        interpreter.close()
        super.onDestroy()
    }
}
