package com.chessreader.plong.application

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.android.example.cameraxbasic.databinding.VibratorActivityBinding
import java.net.ConnectException
import java.net.Socket
import kotlin.concurrent.thread
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.annotation.RequiresApi
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

private const val PERMISSION_REQUEST_CODE = 201
private const val HEADER_SIZE = 5
private const val SIZE_OF_INT = 4
private const val ID_HEADER = "VIBRA"
private const val MOVE_HEADER = "FENVI"
private const val VIBRATION_LENGTH = 200L
private const val TIME_BETWEEN_VIBRATION = 400L
private const val TIME_BETWEEN_SEPARATE_VIBRATION = TIME_BETWEEN_VIBRATION * 2
private const val FIRST_MIN_LETTER_INDEX = 96

class VibratorActivity: AppCompatActivity() {

    private lateinit var binding: VibratorActivityBinding
    private lateinit var preferences: SharedPreferences
    private lateinit var socket: Socket

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = VibratorActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        preferences = getSharedPreferences("myPrefs", MODE_PRIVATE)
        thread { setupSocket() }

        binding.reconnectSocket.setOnClickListener {
            thread {
                setupSocket()
            }
        }

        if (!hasPermissions()) {
            requestPermissions()
        }

        thread {
            for (j in 1..2) {
                for (i in 1..10) {
                    vibrate()
                    Thread.sleep(TIME_BETWEEN_VIBRATION)
                }
                Thread.sleep(TIME_BETWEEN_SEPARATE_VIBRATION)
            }
        }
    }

    /**
     *
     */
    private fun chessToVibrate(move: String) {
        for (letter in move) {
            val value = if (letter.isLetter()) {
                letter.code - FIRST_MIN_LETTER_INDEX
            } else {
                letter.digitToInt()
            }
            for (i in 1.. value) {
                vibrate()
                Thread.sleep(TIME_BETWEEN_VIBRATION)
            }
            Thread.sleep(TIME_BETWEEN_SEPARATE_VIBRATION)
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
        with (socket.getOutputStream()) {
            write(ID_HEADER.toByteArray())
            flush()
        }
        runOnUiThread {
            Toast.makeText(this, "Socket connecté à $address:$port", Toast.LENGTH_SHORT).show()
        }
        binding.connectStatus.isChecked = true

        listenSocket()
    }

    private fun listenSocket() {
        val input = socket.getInputStream()
        while (true) {
            val buffer = ByteArray(HEADER_SIZE)
            input.read(buffer)
            val header = String(buffer)
            if (header != MOVE_HEADER) {
                return closeSocket()
            }
            val sizeBuffer = ByteArray(SIZE_OF_INT)
            input.read(sizeBuffer)
            val size = sizeBuffer.toInt()
            val moveBuffer = ByteArray(size)
            input.read(moveBuffer)
            val move = String(moveBuffer)
            chessToVibrate(move)

            runOnUiThread {
                Toast.makeText(this, "Message reçu: $move", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun closeSocket() {
        socket.close()
        binding.connectStatus.isChecked = false
        Toast.makeText(this, "Fermeture de la socket.", Toast.LENGTH_SHORT).show()
    }

    private fun vibrate() {
        if (Build.VERSION.SDK_INT >= 31) {
            val vibratorManager = applicationContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            val vibrator = vibratorManager.defaultVibrator
            vibrator.vibrate(VibrationEffect.createOneShot(VIBRATION_LENGTH, VibrationEffect.DEFAULT_AMPLITUDE))
        }
        else {
            val v = applicationContext.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            if (Build.VERSION.SDK_INT >= 29) {
                v.vibrate(VibrationEffect.createOneShot(VIBRATION_LENGTH, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                v.vibrate(VIBRATION_LENGTH)
            }
        }
    }

    private fun hasPermissions(): Boolean {
        return ContextCompat.checkSelfPermission(this, android.Manifest.permission.VIBRATE) != PackageManager.PERMISSION_GRANTED
    }

    private fun requestPermissions() {
        ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.VIBRATE), PERMISSION_REQUEST_CODE)
    }

    private fun ByteArray.toInt(): Int {
        var result = 0
        for (i in indices) {
            result = result or (this[i].toInt() shl 8 * i)
        }
        return result
    }
}