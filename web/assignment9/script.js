
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
  button.onclick = processInput;
}
