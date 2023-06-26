const startBtn = document.querySelector("#start-btn");
const resultList = document.querySelector("#result");
const notif = document.querySelector("#notif")

const recognition = new webkitSpeechRecognition();

recognition.continuous = true;
recognition.lang = "en-US";
recognition.interimResults = false;
recognition.maxAlternatives = 1; //only 1

userstring = ""
statements = []
responses = []
index_of_words = 0
function toggleRecording(button) {
    var buttonText = document.getElementById("button-text");
    
    if (buttonText.textContent === "Start Rec") {
        buttonText.textContent = "Stop";
        button.style.backgroundColor = "rgb(106, 75, 59)"; //Brown
        recognition.start();
    } else {
        buttonText.textContent = "Start Rec";
        button.style.backgroundColor = "#529d57"; //Green

        recognition.stop();
    }
  }

startBtn.addEventListener("click", () => {
    toggleRecording(startBtn);
});

recognition.onresult = (e) => {
    recent = e.results[index_of_words][0].transcript;
    userstring += recent;
    statements.push(recent)
    index_of_words++;

    console.log(recent);
    console.log(e)
    console.log(statements)

    const listItem = document.createElement("li");
    listItem.textContent = recent;
    resultList.appendChild(listItem);

    fetch('/useraudiodata', {
    method: 'POST',
    headers: {
    'Content-Type': 'application/json'
    },
    body: JSON.stringify(statements)
    })
    .then(response => response.json())
    .then(responseData => {
    
    console.log(responseData);
    responses.push(responseData)

    const listItem2 = document.createElement("li");
    listItem2.textContent = responseData;
    notif.appendChild(listItem2);

    })
    .catch(error => {
    console.error('Error:', error);
    });
};