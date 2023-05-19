package com.android.example.cameraxbasic

import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.android.example.cameraxbasic.databinding.VibratorActivityBinding
import java.net.ConnectException
import java.net.Socket
import kotlin.concurrent.thread

class VibratorActivity: AppCompatActivity() {

    private lateinit var binding: VibratorActivityBinding
    private lateinit var preferences: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = VibratorActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        preferences = getSharedPreferences("myPrefs", MODE_PRIVATE)
        setupSocket()
    }

    private fun chessToVibrate(string: String) {
        val vibrator = getSystemService(VIBRATOR_SERVICE) as Vibrator
        val lettres = "abcdefgh"
        for(i in string.indices){
            for (j in 0..7){
                if (i == 0 mod 2 && lettres[j]==string[i]){
                    for(k in j){
                    vibrator.vibrate(400)
                }
                    i++
                }
                if(i =! 0 mod 2 && j == StringToInt(string[i])){
                    for(k in j){
                        vibrator.vibrate(400)
                }
                    i++
                    if(i == 2){
                        wait(1000)
                    }
                }
            }
        }

    }

    private fun setupSocket() {
        val address = preferences.getString("address", "0.0.0.0")
        val port = preferences.getInt("port", 8080)

        thread {
            try {
                val socket = Socket(address, port)
                with (socket.getOutputStream()) {
                    write("VIBRA".toByteArray())
                    flush()
                }
                val input = socket.getInputStream()
                while (true) {
                    val buffer = ByteArray(4)
                    input.read(buffer)
                    val move = buffer.toString()
                    runOnUiThread {
                        Toast.makeText(this, "Message reçu: $move", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: ConnectException) {
                runOnUiThread {
                    Toast.makeText(this, "Impossible de se connecter via socket à $address au port $port.", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
}