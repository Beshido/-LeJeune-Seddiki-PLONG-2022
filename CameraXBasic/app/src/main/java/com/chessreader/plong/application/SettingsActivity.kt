package com.chessreader.plong.application

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.android.example.cameraxbasic.databinding.SettingsBinding

/** Fragment used to present the user with a gallery of photos taken */
class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: SettingsBinding
    private lateinit var sharedPreferences: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = SettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // get shared preferences
        sharedPreferences = getSharedPreferences("myPrefs", Context.MODE_PRIVATE)

        // get references to the EditText views and load the saved preferences if any
        binding.adresse.setText(sharedPreferences.getString("address", "0.0.0.0"))
        binding.port.setText(sharedPreferences.getInt("port", 8080).toString())
        binding.socket.isChecked = sharedPreferences.getBoolean("socket", true)

        // get reference to the save button and set a click listener to save the input values in preferences
        binding.saveButton.setOnClickListener {
            with(sharedPreferences.edit()) {
                putString("address", binding.adresse.text.toString())
                putInt("port", binding.port.text.toString().toInt())
                putBoolean("socket", binding.socket.isChecked)
                apply()
            }
            Toast.makeText(this, "Adresse et port indiqués sauvegardés.", Toast.LENGTH_SHORT).show()
        }
    }
}
