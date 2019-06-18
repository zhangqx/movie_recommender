$(function(){
	/*新闻列表加载*/
	$.ajax({
        async: false,　　　　　　　
        type: "get",   
        url: "json/homeNews.json",   
        cache: false,
        dataType: "json",
        success: function(result) {
        	
        	if(result.code == 200){
        		var data = result.data;
        		var imgSrc;
        		var newsClone;
        		var title;
	        	
	            for(var i=0;i<data.length;i++){
	                imgSrc=data[i].imgMobile;
	                title=data[i].title;
	                newsClone=$(".none").clone();
	                newsClone.attr("newId",data[i].id);     /*设置一个自定义属性id 跳转后加载不同数据*/
	                newsClone.removeClass("none");
	                newsClone.find("img").attr("src","upload/"+imgSrc);
					newsClone.find("p").html(title);
	                $(".news_list").append(newsClone);
	        	}	
        	}else{
        		alert("请求失败！")
        	}
        },
        error:function(){
        	alert("异常！")
        }
	});
	$('.news_list').delegate('.news_item', 'click', function () {    //	点击列表
		self.location.href = 'detailNews.html'+"?"+'&newId='+$(this).attr('newId');
	});
	

	
	
})