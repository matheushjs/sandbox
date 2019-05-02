
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

  var url = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    + "?@dataInicial=" + sendData.dataInicial
    + "&@dataFinalCotacao=" + sendData.dataFinalCotacao
    + "&$top=1000"
    + "&$skip=0"
    + "&$orderby=" + sendData.$orderby
    + "&$format=json"
    + "&$select=" + sendData.$select;
  var xhr = new XMLHttpRequest();
  xhr.open("GET", url);
  xhr.responseType = "json";
  xhr.onload = function () {
    var content = document.querySelector("#content");
    content.innerHTML = "";

    var newContent = "<table>";
    newContent += "<thead><tr><th>#</th><th>Cotação</th><th>Data</th></tr></thead>";
    newContent += "<tbody>";
    xhr.response.value.forEach(function(elem, index){
      newContent += "<tr>";
      if(buysell === "buy"){
        newContent += "<td>" + (index+1) + "</td><td>" + elem.cotacaoCompra + "</td><td>" + elem.dataHoraCotacao + "</td>";
      } else {
        newContent += "<td>" + (index+1) + "</td><td>" + elem.cotacaoVenda + "</td><td>" + elem.dataHoraCotacao + "</td>";
      }
      newContent += "</tr>";
    });
    newContent += "</tbody>";
    newContent += "</table>";

    newContent += "<p>Cotação desde: " + sendData.dataInicial + " até o dia de hoje.</p>";

    var div = document.createElement("div");
    div.innerHTML = newContent;

    content.appendChild(div);
  };
  xhr.send();

  event.preventDefault();
}

window.onload = function(){
  var button = document.querySelector("button");
  button.onclick = processInput;
}
