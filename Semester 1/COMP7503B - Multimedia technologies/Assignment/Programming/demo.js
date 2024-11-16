// 读取vacancy.json 和 carpark.json

var fs = require("fs");
var axios = require("axios");

var url_carpark= "http://resource.data.one.gov.hk/td/carpark/basic_info_all.json";
var url_vacancy = "http://resource.data.one.gov.hk/td/carpark/vacancy_all.json";

// 下载到本地 UTF-8 编码
async function downloadJson() {
    var vacancy = await axios.get(url_vacancy);
    var carpark = await axios.get(url_carpark);

    fs.writeFileSync("vacancy_all.json", JSON.stringify(vacancy.data), "utf-8");
    fs.writeFileSync("carpark_all.json", JSON.stringify(carpark.data), "utf-8");
}

// 读取本地文件
async function readJson() {
    var vacancy = JSON.parse(fs.readFileSync("vacancy_all.json", "utf-8"));
    var carpark = JSON.parse(fs.readFileSync("carpark_all.json", "utf-8"));
    
    // 将 vacancy 用 park_id 作为 key 构建字典
    var vacancy_dict = {};
    vacancy["car_park"].forEach(function (v) {
      vacancy_dict[v["park_id"]] = v;
    });
    
    // 根据区域进行分区
    var districts = {};
    var vacancy_data = {};
    
    carpark["car_park"].forEach(function (park) {
      var park_id = park["park_id"];
      var district = park["district_tc"]; // 获取停车场所在区域
    
      var available_spaces = 0;
      if (vacancy_dict[park_id]) {
        available_spaces =
          vacancy_dict[park_id]["vehicle_type"][0]["service_category"][0][
            "vacancy"
          ];
      }
    
      // 按区域进行分组
      if (!districts[district]) {
        districts[district] = [];
        vacancy_data[district] = [];
      }
      districts[district].push(park["name_tc"]);
      vacancy_data[district].push(available_spaces);
    });
    
    // 定义空位区间（例如：0-10，11-20，21-30）
    var bins = [0, 10, 20, 30, 40, 50, 100];
    var bin_labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-100"];
    
    // 生成最终格式的结果数据
    var result = [];
    for (var district in districts) {
      var vacancy_counts = new Array(bins.length - 1).fill(0);
    
      vacancy_data[district].forEach(function (space) {
        for (var i = 0; i < bins.length - 1; i++) {
          if (space >= bins[i] && space < bins[i + 1]) {
            vacancy_counts[i]++;
            break;
          }
        }
      });
    
      // 过滤掉空区间
      var filtered_counts = [];
      var filtered_labels = [];
      for (var i = 0; i < vacancy_counts.length; i++) {
        if (vacancy_counts[i] > 0) {
          filtered_counts.push(vacancy_counts[i]);
          filtered_labels.push(bin_labels[i]);
        }
      }
    
      // 如果有数据，构建符合要求的格式
      if (filtered_counts.length > 0) {
        // 把filtered_counts 处理为 每一个区间是一个数组
        var data = [];
        for (var i = 0; i < filtered_counts.length; i++) {
          data.push([filtered_counts[i]]);
        }
        result.push({
          series: [district], // 这里设置区域名作为 series
          data: data, // 数据
          labels: filtered_labels, // 区间标签
        });
      }
    }
    
    console.log([result[0]]); // 输出第一个区域的数据
    return result;
}

// main 函数
async function main() {
    await downloadJson();
    const result = await readJson();
}

main();
