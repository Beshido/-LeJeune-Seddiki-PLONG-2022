package com.chessreader.plong.application

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.android.example.cameraxbasic.databinding.MenuBinding

class MainMenu internal constructor() : AppCompatActivity() {

    private lateinit var binding: MenuBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = MenuBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.camera.setOnClickListener {
            val intent = Intent(this, PhotoActivity::class.java)
            startActivity(intent)
        }

        binding.vibrateur.setOnClickListener {
            val intent = Intent(this, VibratorActivity::class.java)
            startActivity(intent)
        }

        binding.parametres.setOnClickListener {
            val intent = Intent(this, SettingsActivity::class.java)
            startActivity(intent)
        }
    }
}