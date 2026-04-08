$ErrorActionPreference = "Stop"

function Replace-Text {
    param(
        [string]$Path,
        [string]$Old,
        [string]$New
    )

    $content = Get-Content -Path $Path -Raw
    if (-not $content.Contains($Old)) {
        throw "Expected text not found in $Path"
    }
    $content = $content.Replace($Old, $New)
    [System.IO.File]::WriteAllText($Path, $content)
}

$modelsPath = "C:\Users\Khushi\OneDrive\Desktop\automind_app_frontend\app\src\main\java\com\automind\app\data\model\VehicleModels.kt"
$repoPath = "C:\Users\Khushi\OneDrive\Desktop\automind_app_frontend\app\src\main\java\com\automind\app\data\repository\VehicleRepository.kt"
$profilePath = "C:\Users\Khushi\OneDrive\Desktop\automind_app_frontend\app\src\main\java\com\automind\app\ui\screens\profile\ProfileScreen.kt"
$alertsPath = "C:\Users\Khushi\OneDrive\Desktop\automind_app_frontend\app\src\main\java\com\automind\app\ui\screens\alerts\AlertsScreen.kt"

Replace-Text -Path $modelsPath -Old @'
@JsonClass(generateAdapter = true)
data class ServiceBooking(
    @Json(name = "status") val status: String? = null,
    @Json(name = "booking_id") val bookingId: String? = null,
    @Json(name = "center_name") val centerName: String? = null,
    @Json(name = "center_address") val centerAddress: String? = null,
    @Json(name = "center_phone") val centerPhone: String? = null,
    @Json(name = "distance_km") val distanceKm: Double? = null,
    @Json(name = "eta_minutes") val etaMinutes: Int? = null,
    @Json(name = "urgency") val urgency: String? = null,
    @Json(name = "scheduled_at") val scheduledAt: String? = null,
    @Json(name = "scheduled_date") val scheduledDate: String? = null,
    @Json(name = "scheduled_time") val scheduledTime: String? = null,
    @Json(name = "service_center_lat") val serviceCenterLat: Double? = null,
    @Json(name = "service_center_lon") val serviceCenterLon: Double? = null,
    @Json(name = "vehicle_lat") val vehicleLat: Double? = null,
    @Json(name = "vehicle_lon") val vehicleLon: Double? = null
)
'@ -New @'
@JsonClass(generateAdapter = true)
data class ServiceBooking(
    @Json(name = "status") val status: String? = null,
    @Json(name = "booking_id") val bookingId: String? = null,
    @Json(name = "center_name") val centerName: String? = null,
    @Json(name = "center_address") val centerAddress: String? = null,
    @Json(name = "center_phone") val centerPhone: String? = null,
    @Json(name = "distance_km") val distanceKm: Double? = null,
    @Json(name = "eta_minutes") val etaMinutes: Int? = null,
    @Json(name = "urgency") val urgency: String? = null,
    @Json(name = "scheduled_at") val scheduledAt: String? = null,
    @Json(name = "scheduled_date") val scheduledDate: String? = null,
    @Json(name = "scheduled_time") val scheduledTime: String? = null,
    @Json(name = "requested_date") val requestedDate: String? = null,
    @Json(name = "requested_time") val requestedTime: String? = null,
    @Json(name = "editable") val editable: Boolean? = null,
    @Json(name = "car_id") val carId: String? = null,
    @Json(name = "vehicle_name") val vehicleName: String? = null,
    @Json(name = "service_center_lat") val serviceCenterLat: Double? = null,
    @Json(name = "service_center_lon") val serviceCenterLon: Double? = null,
    @Json(name = "vehicle_lat") val vehicleLat: Double? = null,
    @Json(name = "vehicle_lon") val vehicleLon: Double? = null
)
'@

Replace-Text -Path $modelsPath -Old @'
    val serviceBookingId: String = "",
    val serviceUrgency: String = "",
    val vehicleLat: Double = 0.0,
    val vehicleLon: Double = 0.0
)
'@ -New @'
    val serviceBookingId: String = "",
    val serviceUrgency: String = "",
    val serviceRequestedDate: String = "",
    val serviceRequestedTime: String = "",
    val serviceBookingEditable: Boolean = false,
    val vehicleLat: Double = 0.0,
    val vehicleLon: Double = 0.0
)
'@

Replace-Text -Path $repoPath -Old @'
    private val _uiState = MutableStateFlow(VehicleStateSummary())
    val uiState: StateFlow<VehicleStateSummary> = _uiState.asStateFlow()

    private val _alerts = MutableStateFlow<List<AlertItem>>(emptyList())
    val alerts: StateFlow<List<AlertItem>> = _alerts.asStateFlow()

    private val _recommendation = MutableStateFlow(
'@ -New @'
    private val _uiState = MutableStateFlow(VehicleStateSummary())
    val uiState: StateFlow<VehicleStateSummary> = _uiState.asStateFlow()
    private val stateCache = mutableMapOf<String, VehicleStateSummary>()

    private val _alerts = MutableStateFlow<List<AlertItem>>(emptyList())
    val alerts: StateFlow<List<AlertItem>> = _alerts.asStateFlow()
    private val alertsCache = mutableMapOf<String, List<AlertItem>>()

    private val _recommendation = MutableStateFlow(
'@

Replace-Text -Path $repoPath -Old @'
    val recommendation: StateFlow<RecommendationItem> = _recommendation.asStateFlow()

    private val _isConnected = MutableStateFlow(false)
'@ -New @'
    val recommendation: StateFlow<RecommendationItem> = _recommendation.asStateFlow()
    private val recommendationCache = mutableMapOf<String, RecommendationItem>()

    private val _isConnected = MutableStateFlow(false)
'@

Replace-Text -Path $repoPath -Old @'
    fun setActiveCarId(carId: String) {
        _activeCarId = carId
        Log.d(TAG, "Active car ID set to: $carId")
    }
'@ -New @'
    fun setActiveCarId(carId: String) {
        _activeCarId = carId
        _uiState.value = stateCache[carId] ?: VehicleStateSummary(carId = carId)
        _alerts.value = alertsCache[carId] ?: emptyList()
        _recommendation.value = recommendationCache[carId]
            ?: RecommendationItem(
                "Vehicle reading steady. Continue driving safely.",
                null
            )
        Log.d(TAG, "Active car ID set to: $carId")
    }
'@

Replace-Text -Path $repoPath -Old @'
            serviceBookingId = booking?.bookingId ?: current.serviceBookingId,
            serviceUrgency = booking?.urgency ?: current.serviceUrgency,
            vehicleLat = booking?.vehicleLat ?: recommendation?.vehicleLat ?: obs?.latitude ?: current.vehicleLat,
            vehicleLon = booking?.vehicleLon ?: recommendation?.vehicleLon ?: obs?.longitude ?: current.vehicleLon
        )

        _uiState.value = summary
        generateAlertsAndRecommendations(summary, dashboard?.activeAlerts ?: response.activeAlerts)
    }
'@ -New @'
            serviceBookingId = booking?.bookingId ?: current.serviceBookingId,
            serviceUrgency = booking?.urgency ?: current.serviceUrgency,
            serviceRequestedDate = booking?.requestedDate ?: current.serviceRequestedDate,
            serviceRequestedTime = booking?.requestedTime ?: current.serviceRequestedTime,
            serviceBookingEditable = booking?.editable ?: current.serviceBookingEditable,
            vehicleLat = booking?.vehicleLat ?: recommendation?.vehicleLat ?: obs?.latitude ?: current.vehicleLat,
            vehicleLon = booking?.vehicleLon ?: recommendation?.vehicleLon ?: obs?.longitude ?: current.vehicleLon
        )

        _uiState.value = summary
        stateCache[summary.carId] = summary
        generateAlertsAndRecommendations(summary, dashboard?.activeAlerts ?: response.activeAlerts)
    }
'@

Replace-Text -Path $repoPath -Old @'
            _recommendation.value = RecommendationItem(
                message = topAlert.message ?: "Vehicle health update available.",
                actionText = when {
                    state.serviceBookingStatus != null -> "SERVICE SCHEDULED"
                    state.serviceDueNow -> "Schedule Service"
                    (topAlert.code ?: "").contains("COLLISION", ignoreCase = true) -> "Slow Down"
                    else -> "Inspect Vehicle"
                },
                isCritical = (topAlert.severity ?: "").equals("CRITICAL", ignoreCase = true)
            )
            return
'@ -New @'
            _recommendation.value = RecommendationItem(
                message = topAlert.message ?: "Vehicle health update available.",
                actionText = when {
                    state.serviceBookingStatus != null -> "SERVICE SCHEDULED"
                    state.serviceDueNow -> "Schedule Service"
                    (topAlert.code ?: "").contains("COLLISION", ignoreCase = true) -> "Slow Down"
                    else -> "Inspect Vehicle"
                },
                isCritical = (topAlert.severity ?: "").equals("CRITICAL", ignoreCase = true)
            )
            alertsCache[state.carId] = _alerts.value
            recommendationCache[state.carId] = _recommendation.value
            return
'@

Replace-Text -Path $repoPath -Old @'
        _alerts.value = combined
        _recommendation.value = RecommendationItem(recMsg, action, isCrit)
    }
'@ -New @'
        _alerts.value = combined
        _recommendation.value = RecommendationItem(recMsg, action, isCrit)
        alertsCache[state.carId] = _alerts.value
        recommendationCache[state.carId] = _recommendation.value
    }
'@

Replace-Text -Path $profilePath -Old @'
            items(vehicles) { vehicle ->
                val isLiveVehicle = vehicle.licensePlate == uiState.carId
'@ -New @'
            items(vehicles) { vehicle ->
                val isLiveVehicle = vehicle.isPrimary
'@

Replace-Text -Path $profilePath -Old @'
                            coroutineScope.launch {
                                repository.resetSession(vehicle.licensePlate)
                            }
'@ -New @'
                            coroutineScope.launch {
                                repository.resetSession(
                                    vehicle.licensePlate,
                                    mapOf(
                                        "vehicle_name" to "${vehicle.make} ${vehicle.model} ${vehicle.year}",
                                        "vehicle_maker" to vehicle.make,
                                    )
                                )
                            }
'@

Replace-Text -Path $profilePath -Old @'
                coroutineScope.launch {
                    repository.resetSession(vehicle.licensePlate)
                }
'@ -New @'
                coroutineScope.launch {
                    repository.resetSession(
                        vehicle.licensePlate,
                        mapOf(
                            "vehicle_name" to "${vehicle.make} ${vehicle.model} ${vehicle.year}",
                            "vehicle_maker" to vehicle.make,
                        )
                    )
                }
'@

$alertsContent = Get-Content -Path $alertsPath -Raw

$alertsContent = $alertsContent.Replace(
@'
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
'@,
@'
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
'@
)

$alertsContent = $alertsContent.Replace(
@'
fun AlertsScreen(repository: VehicleRepository) {
    val alerts by repository.alerts.collectAsState()
    val uiState by repository.uiState.collectAsState()
    val coroutineScope = rememberCoroutineScope()
'@,
@'
fun AlertsScreen(repository: VehicleRepository) {
    val alerts by repository.alerts.collectAsState()
    val uiState by repository.uiState.collectAsState()
    val coroutineScope = rememberCoroutineScope()
    var showEditDialog by remember { mutableStateOf(false) }
    var requestedDate by remember { mutableStateOf("") }
    var requestedTime by remember { mutableStateOf("") }

    LaunchedEffect(uiState.serviceScheduledDate, uiState.serviceScheduledTime, uiState.serviceRequestedDate, uiState.serviceRequestedTime) {
        requestedDate = uiState.serviceScheduledDate.ifBlank { uiState.serviceRequestedDate }
        requestedTime = uiState.serviceScheduledTime.ifBlank { uiState.serviceRequestedTime }
    }
'@
)

$alertsContent = $alertsContent.Replace("â€¢", "|")

$alertsContent = $alertsContent.Replace(
@'
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                coroutineScope.launch {
                                    repository.executeAiCycle(
                                        actionType = "request_service",
                                        value = 1.0,
                                        reason = "User requested service from alerts screen"
                                    )
                                }
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = AccentCyan, contentColor = DarkBackground),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text(if (uiState.serviceBookingStatus != null) "SCHEDULED" else "SCHEDULE NOW", fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(Icons.Default.CalendarMonth, contentDescription = null, modifier = Modifier.size(16.dp))
                        }
'@,
@'
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            Button(
                                onClick = {
                                    coroutineScope.launch {
                                        repository.executeAiCycle(
                                            actionType = "request_service",
                                            value = 1.0,
                                            reason = "User requested service from alerts screen"
                                        )
                                    }
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = AccentCyan, contentColor = DarkBackground),
                                shape = RoundedCornerShape(10.dp)
                            ) {
                                Text(if (uiState.serviceBookingStatus != null) "SCHEDULED" else "SCHEDULE NOW", fontWeight = FontWeight.Bold, letterSpacing = 1.sp)
                                Spacer(modifier = Modifier.width(6.dp))
                                Icon(Icons.Default.CalendarMonth, contentDescription = null, modifier = Modifier.size(16.dp))
                            }
                            if (uiState.serviceBookingStatus != null && uiState.serviceBookingEditable) {
                                OutlinedButton(
                                    onClick = { showEditDialog = true },
                                    shape = RoundedCornerShape(10.dp),
                                    colors = ButtonDefaults.outlinedButtonColors(contentColor = AccentCyan)
                                ) {
                                    Text("EDIT", fontWeight = FontWeight.Bold)
                                }
                            }
                        }
'@
)

$dialogBlock = @'

    if (showEditDialog) {
        EditServiceDialog(
            initialDate = requestedDate,
            initialTime = requestedTime,
            onDismiss = { showEditDialog = false },
            onConfirm = { date, time ->
                showEditDialog = false
                coroutineScope.launch {
                    repository.executeAiCycle(
                        actionType = "reschedule_service",
                        value = 1.0,
                        reason = "requested_date=$date;requested_time=$time"
                    )
                }
            }
        )
    }
}

@Composable
private fun EditServiceDialog(
    initialDate: String,
    initialTime: String,
    onDismiss: () -> Unit,
    onConfirm: (String, String) -> Unit,
) {
    var date by remember(initialDate) { mutableStateOf(initialDate) }
    var time by remember(initialTime) { mutableStateOf(initialTime) }

    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = DarkSurface,
        title = { Text("Edit Service Slot", color = TextPrimary, fontWeight = FontWeight.Bold) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = date,
                    onValueChange = { date = it },
                    label = { Text("Preferred Date") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = AccentCyan,
                        unfocusedBorderColor = DarkSurfaceVariant,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedLabelColor = AccentCyan,
                        unfocusedLabelColor = TextSecondary,
                    )
                )
                OutlinedTextField(
                    value = time,
                    onValueChange = { time = it },
                    label = { Text("Preferred Time") },
                    singleLine = true,
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = AccentCyan,
                        unfocusedBorderColor = DarkSurfaceVariant,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedLabelColor = AccentCyan,
                        unfocusedLabelColor = TextSecondary,
                    )
                )
            }
        },
        confirmButton = {
            Button(
                onClick = { onConfirm(date.trim(), time.trim()) },
                colors = ButtonDefaults.buttonColors(containerColor = AccentCyan, contentColor = DarkBackground),
            ) {
                Text("SAVE")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("CANCEL", color = TextSecondary)
            }
        }
    )
}
'@

$alertsContent = [System.Text.RegularExpressions.Regex]::Replace(
    $alertsContent,
    '\n}\s*$',
    $dialogBlock
)

[System.IO.File]::WriteAllText($alertsPath, $alertsContent)
