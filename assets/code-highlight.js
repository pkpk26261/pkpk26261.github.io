/* Shared local syntax highlighting; highlight.js 11.12.0, BSD-3-Clause. */
'use strict';
(() => {
 const labels={auto:'自動辨識',plaintext:'純文字',python:'Python',javascript:'JavaScript',typescript:'TypeScript',c:'C',cpp:'C++',csharp:'C#',java:'Java',bash:'Shell / Bash',xml:'HTML / XML',css:'CSS',json:'JSON',sql:'SQL',yaml:'YAML',swift:'Swift',matlab:'MATLAB'};
 const aliases={js:'javascript',ts:'typescript',html:'xml',sh:'bash',shell:'bash',cs:'csharp','c++':'cpp',py:'python'};
 const escape=text=>String(text).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
 const rendered=new WeakMap();
 function textOf(node){const clone=node.cloneNode(true);clone.querySelectorAll('br').forEach(br=>br.replaceWith('\n'));return clone.textContent||'';}
 function languageOf(figure){const declared=figure.dataset.language;if(declared)return aliases[declared]||declared;const code=figure.querySelector('pre code');const name=(code?.className.match(/language-([\w+-]+)/)||figure.className.match(/highlight\s+([\w+-]+)/))?.[1];return aliases[name]||name||'auto';}
 function highlight(text,language='auto'){
  if(!window.hljs)return {html:escape(text),language};
  try{const result=language==='auto'?hljs.highlightAuto(text,Object.keys(labels).filter(l=>!['auto','plaintext'].includes(l)&&hljs.getLanguage(l))):hljs.highlight(text,{language:hljs.getLanguage(language)?language:'plaintext',ignoreIllegals:true});return {html:result.value,language:result.language||language};}
  catch{return {html:escape(text),language:'plaintext'};}
 }
 function render(root){root.querySelectorAll('figure.highlight').forEach(figure=>{const pre=figure.querySelector('.code pre')||figure.querySelector('pre code')||figure.querySelector('pre');if(!pre)return;const text=textOf(pre),language=languageOf(figure),previous=rendered.get(pre);if(previous?.text===text&&previous?.language===language)return;const result=highlight(text,language);pre.innerHTML=result.html;pre.classList.add('hljs');rendered.set(pre,{text,language});figure.dataset.codeCaption=labels[result.language]||result.language;const caption=figure.querySelector('.code-caption');if(caption)caption.textContent=figure.dataset.codeCaption;});}
 window.StudioCode={labels,textOf,languageOf,highlight,render};
 if(!document.querySelector('.editor-form'))render(document);
})();
