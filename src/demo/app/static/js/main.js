var textarea = document.querySelectorAll('textarea');

for (var i = 0; i < textarea.length; i++) {
    textarea[i].setAttribute('style', 'height:' + (textarea[i].scrollHeight) + 'px;overflow-y:hidden;resize:none')
    textarea[i].addEventListener('input', autosize, false)
}

function autosize() {
    var scrollLeft = window.pageXOffset ||
        (document.documentElement || document.body.parentNode || document.body).scrollLeft;

    var scrollTop = window.pageYOffset ||
        (document.documentElement || document.body.parentNode || document.body).scrollTop;

    this.style.height = 'auto'
    this.style.height = this.scrollHeight + 'px'

    window.scrollTo(scrollLeft, scrollTop)
}

function submitForm() {
    var articleForm = document.querySelector('#form-request')
    var summary = document.querySelector('#summary')
    var articleText = document.querySelector("#article-text").textContent

    articleForm.article.value = articleText
    summary.textContent = "요약 중..."

    articleForm.submit()
}