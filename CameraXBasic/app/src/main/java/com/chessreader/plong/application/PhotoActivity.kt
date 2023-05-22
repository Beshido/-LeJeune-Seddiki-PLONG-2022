package com.chessreader.plong.application

import android.R.attr.bitmap
import android.app.Activity
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts.StartActivityForResult
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.android.example.cameraxbasic.databinding.PhotoActivityBinding
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.ConnectException
import java.net.Socket
import kotlin.concurrent.thread


private const val LICHESS_URL_PREFIX = "https://lichess.org/editor/"
private const val LICHESS_URL_SUFFIX = "?color="
private const val PERMISSION_REQUEST_CODE = 200
private const val HEADER_SIZE = 5
private const val SIZE_OF_INT = 4
private const val ID_HEADER = "PHOTO"
private const val SEND_PHOTO_HEADER = "PICTU"
private const val RECEIVE_TO_FIX_FEN_HEADER = "BOFEN"
private const val SEND_FIXED_FEN_HEADER = "FENSI"
private const val SAVE_PHOTO = true

/**
 * Main entry point into our app. This app follows the single-activity pattern, and all
 * functionality is implemented in the form of fragments.
 */
class PhotoActivity : AppCompatActivity() {

    private lateinit var binding: PhotoActivityBinding
    private lateinit var preferences: SharedPreferences
    private lateinit var socket: Socket
    private lateinit var resultLauncher: ActivityResultLauncher<Intent>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (intent.action.equals(Intent.ACTION_SEND)) {
            if (intent.type.equals("text/plain")) {
                val url = intent.getStringExtra(Intent.EXTRA_TEXT)
                val fen = url?.substringAfter(LICHESS_URL_PREFIX)?.substringBefore(LICHESS_URL_SUFFIX)?.replace("_", " ")
                thread {
                    sendFen(fen!!)
                }
            }
            else if (intent.type!!.startsWith("image/")) {
                val imageUri = intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
                val imageBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
                thread {
                    sendImage(imageBitmap)
                }
            }

        }
        binding = PhotoActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        preferences = getSharedPreferences("myPrefs", MODE_PRIVATE)
        thread { setupSocket() }
        resultLauncher = registerForActivityResult(StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val data: Intent? = result.data
                val imageBitmap = data?.extras?.get("data")
                thread {
                    sendImage(imageBitmap as Bitmap)
                }
            }
        }

        binding.startCamera.setOnClickListener {
            startCamera()
        }

        binding.fenSend.setOnClickListener {
            thread {
                sendFen(binding.fenPrompt.text.toString())
            }
        }

        binding.reconnectSocket.setOnClickListener {
            thread {
                setupSocket()
            }
        }
    }

    private fun setupSocket() {
        val address = preferences.getString("address", "0.0.0.0")
        val port = preferences.getInt("port", 8080)
        try {
            socket = Socket(address, port)
        }
        catch (e: ConnectException) {
            return runOnUiThread {
                Toast.makeText(this, "Impossible de se connecter via socket à $address au port $port.", Toast.LENGTH_SHORT).show()
            }
        }
        socket.getOutputStream().write(ID_HEADER.toByteArray())
        socket.getOutputStream().flush()
        runOnUiThread {
            Toast.makeText(this, "Socket connecté à $address:$port", Toast.LENGTH_SHORT).show()
        }
        binding.connectStatus.isChecked = true

        listenSocket()
    }

    private fun listenSocket() {
        while (true) {
            val inputStream = socket.getInputStream()
            val headerBuffer = ByteArray(HEADER_SIZE)
            inputStream.read(headerBuffer)
            val message = String(headerBuffer)
            if (message != RECEIVE_TO_FIX_FEN_HEADER) {
                return closeSocket()
            }
            val sizeBuffer = ByteArray(SIZE_OF_INT)
            inputStream.read(sizeBuffer)
            val size = sizeBuffer.toInt()
            val fenBuffer = ByteArray(size)
            inputStream.read(fenBuffer)
            val fen = String(fenBuffer).replace(" ", "_")
            val fullUrl = "$LICHESS_URL_PREFIX?fen=$fen"
            runOnUiThread {
                Toast.makeText(this, "Ouverture de $fullUrl", Toast.LENGTH_SHORT).show()
            }
            val browserIntent = Intent(Intent.ACTION_VIEW, Uri.parse(fullUrl))
            startActivity(browserIntent)
        }
    }

    private fun startCamera() {
        if (!hasPermissions()) {
            requestPermissions()
        }
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        resultLauncher.launch(takePictureIntent)
    }

    private fun sendFen(fen: String) {
        val size = fen.length
        with (socket.getOutputStream()) {
            write(SEND_FIXED_FEN_HEADER.toByteArray())
            write(size.toByteArray())
            write(fen.toByteArray())
            flush()
        }
        runOnUiThread {
            Toast.makeText(this, "FEN de $size octets envoyée via socket", Toast.LENGTH_SHORT).show()
        }
    }

    private fun sendImage(data: Bitmap) {
        val buffer = ByteArrayOutputStream()
        data.compress(Bitmap.CompressFormat.PNG, 100, buffer)
        val bufferArray = buffer.toByteArray()
        val size = bufferArray.size
        with (socket.getOutputStream()) {
            write(SEND_PHOTO_HEADER.toByteArray())
            write(size.toByteArray())
            write(bufferArray)
            flush()
        }
        runOnUiThread {
            Toast.makeText(this, "Image de $size octets envoyée via socket", Toast.LENGTH_SHORT).show()
        }
    }

    private fun closeSocket() {
        socket.close()
        binding.connectStatus.isChecked = false
        Toast.makeText(this, "Fermeture de la socket.", Toast.LENGTH_SHORT)
    }

    private fun hasPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                && (!SAVE_PHOTO || (SAVE_PHOTO && ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED))
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), PERMISSION_REQUEST_CODE)
        if (SAVE_PHOTO) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE), PERMISSION_REQUEST_CODE)
        }
    }

    private fun Bitmap.savePicture() {
        val sdCard = Environment.getExternalStorageDirectory()
        val dir = File(sdCard.absolutePath + "/YourFolderName")
        dir.mkdirs()
        val fileName = String.format("%d.jpg", System.currentTimeMillis())
        val outFile = File(dir, fileName)

        val outStream = FileOutputStream(outFile)
        compress(Bitmap.CompressFormat.JPEG, 100, outStream)
        outStream.flush()
        outStream.close()

        val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
        intent.data = Uri.fromFile(outFile)
        sendBroadcast(intent)
    }

    private fun Int.toByteArray(): ByteArray {
        val buffer = ByteArray(4)
        buffer[0] = (this shr 0).toByte()
        buffer[1] = (this shr 8).toByte()
        buffer[2] = (this shr 16).toByte()
        buffer[3] = (this shr 24).toByte()
        return buffer
    }

    private fun ByteArray.toInt(): Int {
        var result = 0
        for (i in indices) {
            result = result or (this[i].toInt() shl 8 * i)
        }
        return result
    }
}
