package com.example.lprtos_client

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.min

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                // Permission is granted. We can now use the camera.
            } else {
                // Explain to the user that the feature is unavailable.
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Check for camera permission
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) -> {
                // You can use the API that requires the permission.
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    VideoStreamingScreen()
                }
            }
        }
    }
}

@Composable
fun VideoStreamingScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture: ListenableFuture<ProcessCameraProvider> = remember { ProcessCameraProvider.getInstance(context) }

    var ipAddress by remember { mutableStateOf(TextFieldValue("0.0.0.0")) }
    var isStreaming by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf("Disconnected") }
    var maskPixels by remember { mutableStateOf<List<PointF>>(emptyList()) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var imageSize by remember { mutableStateOf(Size(0, 0)) }
    var boxSize by remember { mutableStateOf(Size(0, 0)) }
    var rotationDegrees by remember { mutableStateOf(0) }
    var originalImageSize by remember { mutableStateOf(Size(0, 0)) }
    var resolutionText by remember { mutableStateOf("Resolution: N/A") }
    // State for FPS
    var fps by remember { mutableStateOf(0.0) }
    // State for payload size tracking
    var lastPayloadBytes by remember { mutableStateOf(0) }


    val webSocketClient = remember {
        VideoStreamer(
            onStatusUpdate = { newStatus -> status = newStatus },
            onMaskReceived = { points, payloadSize ->
                maskPixels = points
                lastPayloadBytes = payloadSize
            }
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            webSocketClient.disconnect()
        }
    }

    LaunchedEffect(isStreaming) {
        if (isStreaming) {
            webSocketClient.connect(ipAddress.text)
        } else {
            webSocketClient.disconnect()
            maskPixels = emptyList()
            // Reset stats on disconnect
            fps = 0.0
            lastPayloadBytes = 0
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("LPRTOS Client", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text("Status: $status", style = MaterialTheme.typography.bodyLarge)
        Text(resolutionText, style = MaterialTheme.typography.bodyLarge)
        // Display FPS and payload size
        Text("FPS: ${"%.1f".format(fps)}", style = MaterialTheme.typography.bodyLarge)
        Text("Mask payload: ${"%.1f".format(lastPayloadBytes / 1024.0)} KB", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(16.dp))

        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .onSizeChanged {
                    boxSize = Size(it.width, it.height)
                }
        ) {
            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "LPRTOS Client",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            }
            Canvas(modifier = Modifier.fillMaxSize()) {
                if (maskPixels.isNotEmpty() && imageSize.width > 0 && boxSize.width > 0 && originalImageSize.width > 0) {
                    val scaleX = boxSize.width.toFloat() / imageSize.width
                    val scaleY = boxSize.height.toFloat() / imageSize.height
                    val scale = min(scaleX, scaleY)

                    val scaledWidth = imageSize.width * scale
                    val scaledHeight = imageSize.height * scale

                    val offsetX = (boxSize.width - scaledWidth) / 2f
                    val offsetY = (boxSize.height - scaledHeight) / 2f

                    // The server now sends contour points (ordered boundary coordinates)
                    // instead of all interior mask pixels. The drawing logic remains
                    // identical: moveTo first point, lineTo subsequent points, close path.
                    // The visual result is the same outlined mask, but the payload is
                    // 50-100x smaller because only boundary coordinates are transmitted.
                    val path = Path()
                    maskPixels.forEachIndexed { index, pixel ->
                        val transformedX: Float
                        val transformedY: Float

                        when (rotationDegrees) {
                            90 -> {
                                transformedX = pixel.y
                                transformedY = originalImageSize.width - pixel.x
                            }
                            180 -> {
                                transformedX = originalImageSize.width - pixel.x
                                transformedY = originalImageSize.height - pixel.y
                            }
                            270 -> {
                                transformedX = originalImageSize.height - pixel.y
                                transformedY = pixel.x
                            }
                            else -> { // 0 degrees
                                transformedX = pixel.x
                                transformedY = pixel.y
                            }
                        }

                        val rotatedX = imageSize.width - transformedX
                        val rotatedY = imageSize.height - transformedY

                        val finalX = (rotatedX * scale) + offsetX
                        val finalY = (rotatedY * scale) + offsetY

                        if (index == 0) {
                            path.moveTo(finalX, finalY)
                        } else {
                            path.lineTo(finalX, finalY)
                        }
                    }
                    path.close()

                    drawPath(
                        path = path,
                        color = Color.Red.copy(alpha = 0.5f),
                    )
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        OutlinedTextField(
            value = ipAddress,
            onValueChange = { ipAddress = it },
            label = { Text("Enter VM External IP Address") },
            singleLine = true,
            enabled = !isStreaming
        )
        Spacer(Modifier.height(8.dp))

        Button(
            onClick = {
                // Toggle streaming state
                isStreaming = !isStreaming
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isStreaming) "Stop Streaming" else "Start Streaming")
        }
    }

    LaunchedEffect(Unit) {
        val cameraProvider = cameraProviderFuture.get()

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        var cachedMatrix: Matrix? = null
        var lastRotation = -1

        // Variables for FPS calculation
        var frameCount = 0
        var lastFpsUpdateTime = System.currentTimeMillis()

        val imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                    // Drop frame if not streaming
                    if (!isStreaming) {
                        imageProxy.close()
                        return@setAnalyzer
                    }

                    val jpegQuality = 80

                    try {
                        val yuvBytes = imageProxy.toNv21ByteArray()
                        val yuvImage = YuvImage(yuvBytes, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)

                        val jpegOutputStream = ByteArrayOutputStream()
                        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), jpegQuality, jpegOutputStream)
                        val jpegBytes = jpegOutputStream.toByteArray()

                        if (isStreaming) {
                            webSocketClient.sendFrame(jpegBytes)
                        }

                        var displayBitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
                        val currentRotationDegrees = imageProxy.imageInfo.rotationDegrees

                        if (currentRotationDegrees != 0) {
                            if (currentRotationDegrees != lastRotation) {
                                cachedMatrix = Matrix().apply { postRotate(currentRotationDegrees.toFloat()) }
                                lastRotation = currentRotationDegrees
                            }
                            val rotatedBitmap = Bitmap.createBitmap(displayBitmap, 0, 0, displayBitmap.width, displayBitmap.height, cachedMatrix, true)
                            if (rotatedBitmap != displayBitmap) {
                                displayBitmap.recycle()
                            }
                            displayBitmap = rotatedBitmap
                        }

                        CoroutineScope(Dispatchers.Main).launch {
                            bitmap?.recycle()
                            bitmap = displayBitmap
                            imageSize = Size(displayBitmap.width, displayBitmap.height)
                            originalImageSize = Size(imageProxy.width, imageProxy.height)
                            rotationDegrees = currentRotationDegrees
                            resolutionText = "Resolution: ${imageProxy.width} x ${imageProxy.height}"
                        }

                        // --- FPS Calculation ---
                        frameCount++
                        val now = System.currentTimeMillis()
                        val elapsed = now - lastFpsUpdateTime
                        if (elapsed > 1000) { // Update every second
                            val calculatedFps = frameCount * 1000.0 / elapsed
                            CoroutineScope(Dispatchers.Main).launch {
                                fps = calculatedFps
                            }
                            frameCount = 0
                            lastFpsUpdateTime = now
                        }

                    } catch (e: Exception) {
                        Log.e("VideoStreamingScreen", "Error processing frame", e)
                    } finally {
                        imageProxy.close()
                    }
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                imageAnalysis
            )
        } catch (exc: Exception) {
            Log.e("VideoStreamingScreen", "Use case binding failed", exc)
        }
    }
}

class VideoStreamer(
    private val onStatusUpdate: (String) -> Unit,
    private val onMaskReceived: (List<PointF>, Int) -> Unit
) {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    fun connect(ip: String) {
        if (ip.isBlank() || ip.endsWith(".")) {
            onStatusUpdate("Invalid IP Address")
            return
        }
        onStatusUpdate("Connecting...")
        val request = Request.Builder().url("ws://$ip:8765").build()
        val listener = object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                this@VideoStreamer.webSocket = webSocket
                CoroutineScope(Dispatchers.Main).launch { onStatusUpdate("Connected") }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                CoroutineScope(Dispatchers.Main).launch { onStatusUpdate("Failed: ${t.message}") }
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                // Wire format is unchanged: packed little-endian int16 pairs (x, y).
                // With the optimized server, these are contour points (ordered boundary)
                // rather than all interior mask pixels. The parsing logic is identical.
                try {
                    val payloadSize = bytes.size
                    val buffer = bytes.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                    val points = mutableListOf<PointF>()
                    while (buffer.remaining() >= 4) {
                        val x = buffer.getShort()
                        val y = buffer.getShort()
                        points.add(PointF(x.toFloat(), y.toFloat()))
                    }
                    CoroutineScope(Dispatchers.Main).launch {
                        onMaskReceived(points, payloadSize)
                    }
                } catch (e: Exception) {
                    Log.e("VideoStreamer", "Error parsing binary mask data", e)
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                Log.w("VideoStreamer", "Received unexpected text message: $text")
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                CoroutineScope(Dispatchers.Main).launch { onStatusUpdate("Disconnecting") }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                CoroutineScope(Dispatchers.Main).launch { onStatusUpdate("Disconnected") }
            }
        }
        client.newWebSocket(request, listener)
    }

    fun sendFrame(jpegBytes: ByteArray) {
        if (webSocket == null) return
        webSocket?.send(jpegBytes.toByteString())
    }

    fun disconnect() {
        webSocket?.close(1000, "User disconnected")
    }
}

@SuppressLint("UnsafeOptInUsageError")
fun ImageProxy.toNv21ByteArray(): ByteArray {
    val width = this.width
    val height = this.height
    val yBuffer: ByteBuffer = planes[0].buffer
    val uBuffer: ByteBuffer = planes[1].buffer
    val vBuffer: ByteBuffer = planes[2].buffer
    val yRowStride: Int = planes[0].rowStride
    val uRowStride: Int = planes[1].rowStride
    val vRowStride: Int = planes[2].rowStride
    val yPixelStride: Int = planes[0].pixelStride
    val uPixelStride: Int = planes[1].pixelStride
    val vPixelStride: Int = planes[2].pixelStride

    val nv21 = ByteArray(width * height * 3 / 2)
    var yOutIndex = 0
    // Y plane
    for (y in 0 until height) {
        val yInIndex = y * yRowStride
        yBuffer.position(yInIndex)
        yBuffer.get(nv21, yOutIndex, width)
        yOutIndex += width
    }
    // VU plane
    var uvOutIndex = width * height
    for (y in 0 until height / 2) {
        val vInIndex = y * vRowStride
        val uInIndex = y * uRowStride
        vBuffer.position(vInIndex)
        uBuffer.position(uInIndex)
        for (x in 0 until width / 2) {
            val vPixel = vBuffer.get(vInIndex + x * vPixelStride)
            val uPixel = uBuffer.get(uInIndex + x * uPixelStride)
            nv21[uvOutIndex++] = vPixel
            nv21[uvOutIndex++] = uPixel
        }
    }
    return nv21
}
