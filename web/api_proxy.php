<?php
/**
 * JuprisMini-NCM Production PHP Client Proxy
 * Endpoint: http://100.118.242.120:8893/v1/chat
 * Health Check: http://100.118.242.120:8893/health
 */

header('Content-Type: application/json; charset=utf-8');

$ncm_host = '100.118.242.120'; // Perbaikan: tanpa http:// di depan
$ncm_port = 8893;

// Ambil input JSON dari client (frontend index.php)
$raw_input = file_get_contents('php://input');
$data = json_decode($raw_input, true);
$prompt = trim($data['prompt'] ?? '');

if ($prompt === '') {
    echo json_encode([
        'status' => 'error',
        'message' => 'Parameter "prompt" wajib diisi.'
    ], JSON_UNESCAPED_UNICODE);
    exit;
}

// Payload ke Python Daemon NCM
$payload = json_encode([
    'prompt' => $prompt,
    'tool_conf_threshold' => 0.65
]);

$ch = curl_init("http://{$ncm_host}:{$ncm_port}/v1/chat");
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    'Content-Type: application/json',
    'Content-Length: ' . strlen($payload)
]);
curl_setopt($ch, CURLOPT_TIMEOUT, 5);

$response = curl_exec($ch);
$http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
$curl_error = curl_error($ch);
curl_close($ch);

if ($response === false || $http_code !== 200) {
    http_response_code(502);
    echo json_encode([
        'status' => 'error',
        'message' => 'Gagal terhubung ke Jupris-NCM daemon.',
        'details' => $curl_error ?: "HTTP Code: {$http_code}"
    ], JSON_UNESCAPED_UNICODE);
    exit;
}

// Teruskan respons JSON dari NCM daemon ke frontend
echo $response;