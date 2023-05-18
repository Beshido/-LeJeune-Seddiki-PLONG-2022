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
    private lateinit var socket: Socket

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = VibratorActivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        preferences = getSharedPreferences("myPrefs", MODE_PRIVATE)
        setupSocket()
    }

    private fun setupSocket() {
        val address = preferences.getString("address", "0.0.0.0")
        val port = preferences.getInt("port", 8080)

        thread {
            try {
                socket = Socket(address, port)
                with(socket.getOutputStream()) {
                    write("VIBRA".toByteArray())
                    flush()
                }
            } catch (e: ConnectException) {
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Impossible de se connecter via socket à $address au port $port.",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }
}