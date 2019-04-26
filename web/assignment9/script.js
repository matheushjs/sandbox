var cotacoes = [
  {"cotacaoCompra":3.82790,"cotacaoVenda":3.82850,"dataHoraCotacao":"2018-12-03 13:12:23.744"},
  {"cotacaoCompra":3.83070,"cotacaoVenda":3.83130,"dataHoraCotacao":"2018-12-04 13:07:53.025"},
  {"cotacaoCompra":3.85550,"cotacaoVenda":3.85610,"dataHoraCotacao":"2018-12-05 13:12:37.559"},
  {"cotacaoCompra":3.91720,"cotacaoVenda":3.91780,"dataHoraCotacao":"2018-12-06 13:09:35.517"},
  {"cotacaoCompra":3.89580,"cotacaoVenda":3.89640,"dataHoraCotacao":"2018-12-07 13:10:35.035"},
  {"cotacaoCompra":3.91040,"cotacaoVenda":3.91100,"dataHoraCotacao":"2018-12-10 13:09:51.141"},
  {"cotacaoCompra":3.90070,"cotacaoVenda":3.90130,"dataHoraCotacao":"2018-12-11 13:09:47.641"},
  {"cotacaoCompra":3.86230,"cotacaoVenda":3.86290,"dataHoraCotacao":"2018-12-12 13:04:48.141"},
  {"cotacaoCompra":3.87840,"cotacaoVenda":3.87900,"dataHoraCotacao":"2018-12-13 13:05:17.189"},
  {"cotacaoCompra":3.90840,"cotacaoVenda":3.90900,"dataHoraCotacao":"2018-12-14 13:06:56.641"},
  {"cotacaoCompra":3.91150,"cotacaoVenda":3.91210,"dataHoraCotacao":"2018-12-17 13:08:43.268"},
  {"cotacaoCompra":3.89910,"cotacaoVenda":3.89970,"dataHoraCotacao":"2018-12-18 13:11:50.052"},
  {"cotacaoCompra":3.89010,"cotacaoVenda":3.89070,"dataHoraCotacao":"2018-12-19 13:04:57.292"},
  {"cotacaoCompra":3.84370,"cotacaoVenda":3.84430,"dataHoraCotacao":"2018-12-20 13:03:35.898"},
  {"cotacaoCompra":3.86650,"cotacaoVenda":3.86710,"dataHoraCotacao":"2018-12-21 13:07:13.892"},
  {"cotacaoCompra":3.88390,"cotacaoVenda":3.88550,"dataHoraCotacao":"2018-12-24 11:05:11.875"},
  {"cotacaoCompra":3.92520,"cotacaoVenda":3.92580,"dataHoraCotacao":"2018-12-26 13:05:34.012"},
  {"cotacaoCompra":3.93240,"cotacaoVenda":3.93300,"dataHoraCotacao":"2018-12-27 13:07:19.149"},
  {"cotacaoCompra":3.87420,"cotacaoVenda":3.87480,"dataHoraCotacao":"2018-12-28 13:03:50.793"},
  {"cotacaoCompra":3.87420,"cotacaoVenda":3.87480,"dataHoraCotacao":"2018-12-31 11:04:06.627"},
  {"cotacaoCompra":3.85890,"cotacaoVenda":3.85950,"dataHoraCotacao":"2019-01-02 13:04:46.568"},
  {"cotacaoCompra":3.76770,"cotacaoVenda":3.76830,"dataHoraCotacao":"2019-01-03 13:04:50.817"},
  {"cotacaoCompra":3.76210,"cotacaoVenda":3.76270,"dataHoraCotacao":"2019-01-04 13:06:29.332"},
  {"cotacaoCompra":3.70560,"cotacaoVenda":3.70620,"dataHoraCotacao":"2019-01-07 13:09:39.652"},
  {"cotacaoCompra":3.72020,"cotacaoVenda":3.72080,"dataHoraCotacao":"2019-01-08 13:09:06.45"},
  {"cotacaoCompra":3.69250,"cotacaoVenda":3.69310,"dataHoraCotacao":"2019-01-09 13:22:28.243"},
  {"cotacaoCompra":3.68630,"cotacaoVenda":3.68690,"dataHoraCotacao":"2019-01-10 13:03:05.634"},
  {"cotacaoCompra":3.71350,"cotacaoVenda":3.71410,"dataHoraCotacao":"2019-01-11 13:09:51.933"},
  {"cotacaoCompra":3.72550,"cotacaoVenda":3.72600,"dataHoraCotacao":"2019-01-14 13:07:34.594"},
  {"cotacaoCompra":3.70430,"cotacaoVenda":3.70490,"dataHoraCotacao":"2019-01-15 13:13:35.572"},
  {"cotacaoCompra":3.71910,"cotacaoVenda":3.71970,"dataHoraCotacao":"2019-01-16 13:07:34.226"},
  {"cotacaoCompra":3.75850,"cotacaoVenda":3.75910,"dataHoraCotacao":"2019-01-17 13:11:05.484"},
  {"cotacaoCompra":3.74800,"cotacaoVenda":3.74860,"dataHoraCotacao":"2019-01-18 13:10:12.53"},
  {"cotacaoCompra":3.76990,"cotacaoVenda":3.77050,"dataHoraCotacao":"2019-01-21 13:05:38.72"},
  {"cotacaoCompra":3.76090,"cotacaoVenda":3.76150,"dataHoraCotacao":"2019-01-22 13:11:32.497"},
  {"cotacaoCompra":3.79880,"cotacaoVenda":3.79940,"dataHoraCotacao":"2019-01-23 13:07:25.966"},
  {"cotacaoCompra":3.78090,"cotacaoVenda":3.78150,"dataHoraCotacao":"2019-01-24 13:08:47.398"},
  {"cotacaoCompra":3.76130,"cotacaoVenda":3.76260,"dataHoraCotacao":"2019-01-25 13:05:43.204"},
  {"cotacaoCompra":3.76700,"cotacaoVenda":3.76760,"dataHoraCotacao":"2019-01-28 13:08:36.868"},
  {"cotacaoCompra":3.73640,"cotacaoVenda":3.73700,"dataHoraCotacao":"2019-01-29 13:19:36.769"},
  {"cotacaoCompra":3.71450,"cotacaoVenda":3.71510,"dataHoraCotacao":"2019-01-30 13:08:47.599"},
  {"cotacaoCompra":3.65130,"cotacaoVenda":3.65190,"dataHoraCotacao":"2019-01-31 13:02:55.88"},
  {"cotacaoCompra":3.66880,"cotacaoVenda":3.66940,"dataHoraCotacao":"2019-02-01 13:09:03.898"},
  {"cotacaoCompra":3.67500,"cotacaoVenda":3.67560,"dataHoraCotacao":"2019-02-04 13:11:36.952"},
  {"cotacaoCompra":3.67350,"cotacaoVenda":3.67410,"dataHoraCotacao":"2019-02-05 13:10:43.043"},
  {"cotacaoCompra":3.70130,"cotacaoVenda":3.70190,"dataHoraCotacao":"2019-02-06 13:11:47.946"},
  {"cotacaoCompra":3.71870,"cotacaoVenda":3.71930,"dataHoraCotacao":"2019-02-07 13:12:02.413"},
  {"cotacaoCompra":3.71780,"cotacaoVenda":3.71840,"dataHoraCotacao":"2019-02-08 13:03:36.561"},
  {"cotacaoCompra":3.73850,"cotacaoVenda":3.73910,"dataHoraCotacao":"2019-02-11 13:10:41.426"},
  {"cotacaoCompra":3.72900,"cotacaoVenda":3.72960,"dataHoraCotacao":"2019-02-12 13:09:34.609"},
  {"cotacaoCompra":3.72710,"cotacaoVenda":3.72770,"dataHoraCotacao":"2019-02-13 13:06:42.606"},
  {"cotacaoCompra":3.77500,"cotacaoVenda":3.77560,"dataHoraCotacao":"2019-02-14 13:07:55.499"},
  {"cotacaoCompra":3.71490,"cotacaoVenda":3.71550,"dataHoraCotacao":"2019-02-15 13:03:29.581"},
  {"cotacaoCompra":3.73100,"cotacaoVenda":3.73160,"dataHoraCotacao":"2019-02-18 13:10:48.574"},
  {"cotacaoCompra":3.72000,"cotacaoVenda":3.72060,"dataHoraCotacao":"2019-02-19 13:09:51.281"},
  {"cotacaoCompra":3.70940,"cotacaoVenda":3.71000,"dataHoraCotacao":"2019-02-20 13:10:38.213"},
  {"cotacaoCompra":3.75890,"cotacaoVenda":3.75950,"dataHoraCotacao":"2019-02-21 13:11:30.917"},
  {"cotacaoCompra":3.74240,"cotacaoVenda":3.74300,"dataHoraCotacao":"2019-02-22 13:04:45.851"},
  {"cotacaoCompra":3.72790,"cotacaoVenda":3.72850,"dataHoraCotacao":"2019-02-25 13:04:32.81"},
  {"cotacaoCompra":3.75890,"cotacaoVenda":3.75950,"dataHoraCotacao":"2019-02-26 13:02:36.426"},
  {"cotacaoCompra":3.73450,"cotacaoVenda":3.73510,"dataHoraCotacao":"2019-02-27 13:10:37.626"},
  {"cotacaoCompra":3.73790,"cotacaoVenda":3.73850,"dataHoraCotacao":"2019-02-28 13:07:41.798"},
  {"cotacaoCompra":3.78260,"cotacaoVenda":3.78320,"dataHoraCotacao":"2019-03-01 13:25:14.449"},
  {"cotacaoCompra":3.82970,"cotacaoVenda":3.83030,"dataHoraCotacao":"2019-03-06 15:41:49.299"},
  {"cotacaoCompra":3.84810,"cotacaoVenda":3.84870,"dataHoraCotacao":"2019-03-07 13:10:11.6"},
  {"cotacaoCompra":3.86720,"cotacaoVenda":3.86780,"dataHoraCotacao":"2019-03-08 13:02:35.458"},
  {"cotacaoCompra":3.84550,"cotacaoVenda":3.84610,"dataHoraCotacao":"2019-03-11 13:05:37.826"},
  {"cotacaoCompra":3.81230,"cotacaoVenda":3.81290,"dataHoraCotacao":"2019-03-12 13:02:32.417"},
  {"cotacaoCompra":3.82590,"cotacaoVenda":3.82650,"dataHoraCotacao":"2019-03-13 13:07:32.528"},
  {"cotacaoCompra":3.83210,"cotacaoVenda":3.83270,"dataHoraCotacao":"2019-03-14 13:04:49.965"},
  {"cotacaoCompra":3.83380,"cotacaoVenda":3.83440,"dataHoraCotacao":"2019-03-15 13:08:41.666"},
  {"cotacaoCompra":3.81050,"cotacaoVenda":3.81110,"dataHoraCotacao":"2019-03-18 13:10:41.916"},
  {"cotacaoCompra":3.77560,"cotacaoVenda":3.77620,"dataHoraCotacao":"2019-03-19 13:06:32.31"},
  {"cotacaoCompra":3.78910,"cotacaoVenda":3.78970,"dataHoraCotacao":"2019-03-20 13:08:41.147"},
  {"cotacaoCompra":3.79610,"cotacaoVenda":3.79670,"dataHoraCotacao":"2019-03-21 13:04:01.75"},
  {"cotacaoCompra":3.88090,"cotacaoVenda":3.88150,"dataHoraCotacao":"2019-03-22 13:03:37.9"},
  {"cotacaoCompra":3.87640,"cotacaoVenda":3.87700,"dataHoraCotacao":"2019-03-25 13:05:19.849"},
  {"cotacaoCompra":3.86400,"cotacaoVenda":3.86460,"dataHoraCotacao":"2019-03-26 13:10:30.521"},
  {"cotacaoCompra":3.93830,"cotacaoVenda":3.93890,"dataHoraCotacao":"2019-03-27 13:04:31.363"},
  {"cotacaoCompra":3.96760,"cotacaoVenda":3.96820,"dataHoraCotacao":"2019-03-28 13:05:48.911"},
  {"cotacaoCompra":3.89610,"cotacaoVenda":3.89670,"dataHoraCotacao":"2019-03-29 13:05:04.372"},
  {"cotacaoCompra":3.86760,"cotacaoVenda":3.86820,"dataHoraCotacao":"2019-04-01 13:11:37.13"},
  {"cotacaoCompra":3.86550,"cotacaoVenda":3.86610,"dataHoraCotacao":"2019-04-02 13:06:40.208"},
  {"cotacaoCompra":3.84300,"cotacaoVenda":3.84360,"dataHoraCotacao":"2019-04-03 13:08:42.683"},
  {"cotacaoCompra":3.87070,"cotacaoVenda":3.87130,"dataHoraCotacao":"2019-04-04 13:04:33.916"},
  {"cotacaoCompra":3.86160,"cotacaoVenda":3.86220,"dataHoraCotacao":"2019-04-05 13:10:41.534"},
  {"cotacaoCompra":3.86520,"cotacaoVenda":3.86580,"dataHoraCotacao":"2019-04-08 13:02:58.414"},
  {"cotacaoCompra":3.85570,"cotacaoVenda":3.85630,"dataHoraCotacao":"2019-04-09 13:02:35.147"},
  {"cotacaoCompra":3.83390,"cotacaoVenda":3.83450,"dataHoraCotacao":"2019-04-10 13:04:52.338"},
  {"cotacaoCompra":3.83930,"cotacaoVenda":3.83990,"dataHoraCotacao":"2019-04-11 13:07:38.923"},
  {"cotacaoCompra":3.86790,"cotacaoVenda":3.86850,"dataHoraCotacao":"2019-04-12 13:09:35.193"},
  {"cotacaoCompra":3.87240,"cotacaoVenda":3.87300,"dataHoraCotacao":"2019-04-15 13:04:48.792"},
  {"cotacaoCompra":3.89070,"cotacaoVenda":3.89130,"dataHoraCotacao":"2019-04-16 13:11:50.933"},
  {"cotacaoCompra":3.92190,"cotacaoVenda":3.92250,"dataHoraCotacao":"2019-04-17 13:06:34.829"},
  {"cotacaoCompra":3.93640,"cotacaoVenda":3.93700,"dataHoraCotacao":"2019-04-18 13:09:54.238"},
  {"cotacaoCompra":3.92240,"cotacaoVenda":3.92300,"dataHoraCotacao":"2019-04-22 13:09:34.804"},
  {"cotacaoCompra":3.94300,"cotacaoVenda":3.94360,"dataHoraCotacao":"2019-04-23 13:08:45.722"},
  {"cotacaoCompra":3.96240,"cotacaoVenda":3.96300,"dataHoraCotacao":"2019-04-24 13:03:31.357"},
  {"cotacaoCompra":3.97190,"cotacaoVenda":3.97250,"dataHoraCotacao":"2019-04-25 13:10:33.168"},
  {"cotacaoCompra":3.93470,"cotacaoVenda":3.93530,"dataHoraCotacao":"2019-04-26 13:11:30.05"}
];
cotacoes.reverse();

function processInputNoAJAX(event){
  var period = document.querySelector("#period").value;
  var buysell = document.querySelector("#buysell").value;
  var order = document.querySelector("#order").value;

  // Get only from the given periof
  cotacoes2 = cotacoes.filter(function(elem, index){
    return index < period;
  });

  if(buysell === "buy" && order === "asc"){
    cotacoes2.sort(function(a, b){ return a.cotacaoCompra > b.cotacaoCompra; });
  } else if(buysell === "buy" && order === "desc"){
    cotacoes2.sort(function(a, b){ return a.cotacaoCompra < b.cotacaoCompra; });
  } else if(buysell === "sell" && order === "asc"){
    cotacoes2.sort(function(a, b){ return a.cotacaoVenda > b.cotacaoVenda; });
  } else if(buysell === "sell" && order === "desc"){
    cotacoes2.sort(function(a, b){ return a.cotacaoVenda < b.cotacaoVenda; });
  } else {
    console.err("Something went wrong.");
    return;
  }

  arr = []
  if(buysell === "buy"){
    cotacoes2.forEach(function(elem){
      arr.push({"cotacao": elem.cotacaoCompra, "data": elem.dataHoraCotacao});
    });
  } else if(buysell === "sell"){
    cotacoes2.forEach(function(elem){
      arr.push({"cotacao": elem.cotacaoVenda, "data": elem.dataHoraCotacao});
    });
  } else {
    console.err("Something went wrong.");
  }

  var content = document.querySelector("#content");
  content.innerHTML = "";

  var newContent = "<table>";
  newContent += "<thead><tr><th>#</th><th>Cotação</th><th>Data</th></tr></thead>";
  newContent += "<tbody>";
  arr.forEach(function(elem, index){
    newContent += "<tr>";
    newContent += "<td>" + (index+1) + "</td><td>" + elem.cotacao + "</td><td>" + elem.data + "</td>";
    newContent += "</tr>";
  });
  newContent += "</tbody>";
  newContent += "</table>";

  var div = document.createElement("div");
  div.innerHTML = newContent;

  content.appendChild(div);

  event.preventDefault();
}


function processInput(event){
  var period = document.querySelector("#period").value;
  var buysell = document.querySelector("#buysell").value;
  var order = document.querySelector("#order").value;

  // Object we will need to send to the bank
  var sendData = {
    dataInicial: null,
    dataFinalCotacao: null,
    $top: 1000,
    $skip: 0,
    $orderby: "cotacaoCompra desc",
    $format: "json",
    $select: "cotacaoCompra,cotacaoVenda,dataHoraCotacao",
  };

  var now = new Date(Date.now());
  var dateEnd = new Date(2019, now.getMonth(), now.getDate()); // Data final
  sendData.dataFinalCotacao = "\'" + dateEnd.getMonth() + "-" + dateEnd.getDate() + "-" + dateEnd.getFullYear() + "\'";

  // 'period' is the number of days
  // We need it to convert in number of milliseconds
  var milli = period * 24 * 60 * 60 * 1000;
  var past = new Date(Date.now() - milli);
  var dateBegin= new Date(2019, past.getMonth(), past.getDate());
  sendData.dataInicial = "\'" + dateBegin.getMonth() + "-" + dateBegin.getDate() + "-" + dateBegin.getFullYear() + "\'";

  if(buysell === "buy" && order === "asc"){
    sendData.$orderby = "cotacaoCompra asc";
  } else if(buysell === "buy" && order === "desc"){
    sendData.$orderby = "cotacaoCompra desc";
  } else if(buysell === "sell" && order === "asc"){
    sendData.$orderby = "cotacaoVenda asc";
  } else if(buysell === "sell" && order === "desc"){
    sendData.$orderby = "cotacaoVenda desc";
  } else {
    console.err("Something went wrong.");
    return;
  }

  if(buysell === "buy"){
    sendData.$select = "cotacaoCompra,dataHoraCotacao";
  } else if(buysell === "sell"){
    sendData.$select = "cotacaoVenda,dataHoraCotacao";
  } else {
    console.err("Something went wrong.");
  }

  var url = "http://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    + "?@dataInicial=" + sendData.dataInicial
    + "&@dataFinalCotacao=" + sendData.dataFinalCotacao
    + "&$top=1000"
    + "&$skip=0"
    + "&$orderby=" + sendData.$orderby
    + "&$format=json"
    + "&$select=" + sendData.$select;
  console.log(url);
  var xhr = new XMLHttpRequest();
  xhr.open("GET", escape(url));
  xhr.responseType = "json";
  xhr.onreadystatechange = function () {
    if(xhr.readyState === 4 && xhr.status === 200) {
      console.log(xhr.responseText);
    }
  };
  xhr.send();

  console.log(sendData);

  event.preventDefault();
}

window.onload = function(){
  var button = document.querySelector("button");
  button.onclick = processInputNoAJAX;
}
