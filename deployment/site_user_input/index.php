<!DOCTYPE html>
<html>
<head>
    <title>Topics Prediction Form</title>
</head>
<body>
    <h2>Topics Prediction</h2>
    <form action="https://www.kiwinokoto.com.5000/predict" method="post">
        <label for="content">Enter your content:</label><br>
        <textarea id="content" name="content" rows="4" cols="50"></textarea><br><br>
        <input type="submit" value="Predict Topics">
    </form>

    <?php
    // Display the result if available
    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        $content = $_POST['content'];
        $url = "http://127.0.0.1:5000/predict";
        $data = array('content' => $content);

        $options = array(
            'http' => array(
                'header' => "Content-type: application/x-www-form-urlencoded\r\n",
                'method' => 'POST',
                'content' => http_build_query($data),
            ),
        );

        $context = stream_context_create($options);
        $result = file_get_contents($url, false, $context);

        if ($result === FALSE) {
            echo "Error while fetching prediction result.";
        } else {
            $result_data = json_decode($result, true);
            if (isset($result_data['result'])) {
                echo "<h3>Predicted Topics:</h3>";
                echo "<ul>";
                foreach ($result_data['result'] as $topic) {
                    echo "<li>$topic</li>";
                }
                echo "</ul>";
            } elseif (isset($result_data['error'])) {
                echo "<p>Error: " . $result_data['error'] . "</p>";
            }
        }
    }
    ?>
</body>
</html>
