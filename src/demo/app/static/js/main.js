var textarea = document.querySelector('textarea');

autosize(textarea);
textarea.addEventListener('keydown', function(){autosize(this)});

function autosize(el){
    el.style.cssText = 'height:auto; padding:0';
    var a = el.scrollHeight > 75 ? el.scrollHeight + 75 : 150;
    // for box-sizing other than "content-box" use:
    // el.style.cssText = '-moz-box-sizing:content-box';
    el.style.cssText = 'height:' + a + 'px';
}